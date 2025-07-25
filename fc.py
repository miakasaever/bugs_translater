import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import re
from math import log

USE_SBERT = True
RETRIEVAL_METHOD = 'BM25'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size
        self.vocab = set()
        self.doc_freqs = []
        self.idf = {}
        self.build_index()

    def build_index(self):
        for document in self.corpus:
            self.vocab.update(document)

        self.doc_freqs = [defaultdict(int) for _ in range(self.corpus_size)]
        for i, document in enumerate(self.corpus):
            for word in document:
                self.doc_freqs[i][word] += 1

        for word in self.vocab:
            doc_count = sum(1 for i in range(self.corpus_size) if word in self.doc_freqs[i])
            self.idf[word] = log((self.corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        query_words = set(query)

        for i in range(self.corpus_size):
            doc = self.corpus[i]
            doc_len = len(doc)
            for word in query_words:
                if word not in self.doc_freqs[i]:
                    continue

                tf = self.doc_freqs[i][word]
                idf = self.idf[word]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf * numerator / denominator

        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score > 0:
            scores = (scores - min_score) / (max_score - min_score)
        return scores




class EnhancedDictionary:
    def __init__(self, dictionary_path):
        self.dictionary = self.load_dictionary(dictionary_path)
        self.word_index = self.build_word_index()

    def load_dictionary(self, path):
        """
        :param path:dict path
        :return: dict
        """
        dictionary = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    dictionary.append(json.loads(line))
            return dictionary
        except Exception as e:
            return []

    def build_word_index(self):
        """
        :return:
        """
        word_index = defaultdict(list)
        for entry in self.dictionary:
            za_word = entry['za_word'].strip()
            word_index[za_word].append(entry)
            if ' ' in za_word:
                word_index[za_word.replace(' ', '')].append(entry)
        return word_index

    def lookup(self, word):
        """

        """
        clean_word = word.strip()
        for word_form in [clean_word, clean_word.replace(' ', '')]:
            if word_form in self.word_index:
                return self.word_index[word_form]
        return []


class ParallelCorpusRetriever:
    def __init__(self, corpus_path,retrieval_method=RETRIEVAL_METHOD):
        self.corpus = self.load_corpus(corpus_path)
        self.vectorizer = None
        #self.vectorizer=TfidfVectorizer()
        self.retrieval_method=retrieval_method
        self.bm25=None

        self.semantic_model = None
        self.semantic_embeddings = None
        self.model_loaded = False
        if self.corpus:
            self.build_index()

    def load_corpus(self, path):
        """
        :param path:corpus path
        :return corpus
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return []

    def set_semantic_model(self, model):
        """
        设置语义模型
        """
        self.semantic_model = model
        self.model_loaded = model is not None

    def build_index(self):
        """
        构建索引
        """
        self.corpus_texts = [item['za'] for item in self.corpus]

        if self.retrieval_method == 'TFIDF':
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)
        elif self.retrieval_method == 'BM25':

            count_vectorizer = CountVectorizer()
            count_matrix = count_vectorizer.fit_transform(self.corpus_texts)

            tokenized_corpus = []
            for i in range(count_matrix.shape[0]):
                _, cols = count_matrix[i].nonzero()
                words = [count_vectorizer.get_feature_names_out()[col] for col in cols]
                tokenized_corpus.append(words)

            self.bm25 = BM25(tokenized_corpus)

    def find_top_examples(self, query, top_k=2):
        if not self.corpus:
            return []

        if self.retrieval_method == 'TFIDF':
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        elif self.retrieval_method == 'BM25':
            # 对查询进行分词
            count_vectorizer = CountVectorizer()
            count_vectorizer.fit_transform([query])
            tokenized_query = count_vectorizer.get_feature_names_out()
            scores = self.bm25.get_scores(tokenized_query)

        if self.model_loaded and self.semantic_embeddings is not None:
            try:
                query_embedding = self.semantic_model.encode([query])
                semantic_sim = cosine_similarity(query_embedding, self.semantic_embeddings)[0]
                if self.retrieval_method == 'TFIDF':
                    combined_scores = 0.3 * scores + 0.7 * semantic_sim
                else:
                    combined_scores = 0.4 * scores + 0.6 * semantic_sim
            except Exception as e:
                combined_scores = scores
        else:
            combined_scores = scores

        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [(self.corpus[idx], combined_scores[idx]) for idx in top_indices]


class EnhancedGrammarBookRetriever:
    def __init__(self, grammar_book_path,retrieval_method=RETRIEVAL_METHOD):
        self.grammar_book = self.load_grammar_book(grammar_book_path)
        #self.vectorizer = TfidfVectorizer()
        self.vectorizer=None
        self.bm25=None
        self.semantic_model = None
        self.semantic_embeddings = None
        self.model_loaded = False
        self.retrieval_method=retrieval_method

        if USE_SBERT:
            self.load_semantic_model()

        self.build_index()

    def load_grammar_book(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise

    def load_semantic_model(self):
        """
        加载语义模型
        """
        try:
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            cache_dir = "./model_cache"

            local_path = os.path.join(cache_dir, model_name.replace("/", "_"))
            if os.path.exists(local_path):
                self.semantic_model = SentenceTransformer(local_path)
                self.model_loaded = True
            else:
                os.makedirs(cache_dir, exist_ok=True)
                self.semantic_model = SentenceTransformer(
                    f'sentence-transformers/{model_name}',
                    cache_folder=cache_dir
                )
                self.model_loaded = True
        except Exception as e:
            self.semantic_model = None
            self.model_loaded = False

    def build_index(self):
        """
        构建索引
        """
        self.rule_texts = []
        self.structured_features = []

        for rule in self.grammar_book:
            full_text = rule.get('grammar_description', '') + " "
            for example in rule.get('examples', []):
                full_text += example.get('za', '') + " "
                full_text += example.get('zh', '') + " "

                for word, expl in example.get('related_words', {}).items():
                    full_text += f"{word}:{expl} "
            self.rule_texts.append(full_text.strip())

            self.structured_features.append({
                "keywords": list(rule.get('examples', {})),
                "description": rule.get('grammar_description', "")
            })

        if self.retrieval_method == 'TFIDF':
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.rule_texts)
        elif self.retrieval_method == 'BM25':
            count_vectorizer = CountVectorizer()
            count_matrix = count_vectorizer.fit_transform(self.rule_texts)

            tokenized_corpus = []
            for i in range(count_matrix.shape[0]):

                _, cols = count_matrix[i].nonzero()
                words = [count_vectorizer.get_feature_names_out()[col] for col in cols]
                tokenized_corpus.append(words)

            self.bm25 = BM25(tokenized_corpus)

        if self.model_loaded:
            try:
                semantic_texts = [
                    f" {feat['description']} {' '.join(feat['keywords'])}"
                    for feat in self.structured_features
                ]
                self.semantic_embeddings = self.semantic_model.encode(semantic_texts)
            except Exception as e:
                self.semantic_embeddings = None
                self.model_loaded = False

    def find_top_rules(self, query, top_k=3):
        if self.retrieval_method == 'TFIDF':
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        elif self.retrieval_method == 'BM25':
            count_vectorizer = CountVectorizer()
            count_vectorizer.fit_transform([query])
            tokenized_query = count_vectorizer.get_feature_names_out()
            scores = self.bm25.get_scores(tokenized_query)

        if self.model_loaded and self.semantic_embeddings is not None:
            try:
                query_embedding = self.semantic_model.encode([query])
                semantic_sim = cosine_similarity(query_embedding, self.semantic_embeddings)[0]
                if self.retrieval_method == 'TFIDF':
                    combined_scores = 0.25 * scores + 0.75 * semantic_sim
                else:
                    combined_scores = 0.35 * scores + 0.65 * semantic_sim
            except Exception as e:
                combined_scores = scores
        else:
            combined_scores = scores

        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [(self.grammar_book[idx], combined_scores[idx]) for idx in top_indices]


class PromptConstructor:
    @staticmethod
    def format_rule(rule):
        """
        格式化语法规则用于prompt
        """
        formatted = f"规则描述: {rule.get('grammar_description', '')}\n"

        examples = rule.get('examples', [])
        if examples:
            for ex in examples:
                formatted += f"示例: {ex.get('za', '')} → {ex.get('zh', '')}\n"

        return formatted

    @staticmethod
    def format_dictionary_entries(entries):
        """
        格式化词典条目
        """
        if not entries:
            return ""

        formatted = "## 词汇释义:\n"
        for entry in entries:
            formatted += f"- '{entry['za_word']}参考释义': "
            if 'zh_meanings_full' in entry and entry['zh_meanings_full']:
                meanings = ", ".join([m.split('[')[0].strip() for m in entry['zh_meanings_full'][:]])
                formatted += f"{meanings}\n"
            elif 'zh_meanings' in entry and entry['zh_meanings']:
                meanings = ", ".join(entry['zh_meanings'][:])
                formatted += f"{meanings}\n"
        return formatted

    @staticmethod
    def format_example(example):
        """
        parallel corpus
        """
        return f"壮语: {example['za']}\n汉语: {example['zh']}\n"

    @staticmethod
    def extract_paragraph_id(sentence_id):
        """
        从句子ID中提取段落ID
        """
        match = re.match(r'([a-z]+\d+)-', sentence_id)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def construct_prompt(test_item, top_rules, dictionary_entries, top_examples):
        """
        构造最终prompt
        """
        # 提取段落ID用于上下文
        paragraph_id = PromptConstructor.extract_paragraph_id(test_item['id'])

        prompt = (
            "你是一位壮语翻译专家，请根据提供的上下文信息、语法规则、词汇释义和翻译示例，"
            "将下面的壮语句子翻译成汉语。请严格遵守以下要求：\n"
            "1. 保持壮语基本语序（主语-动词-宾语）\n"
            "2. 修饰语后置于被修饰词\n"
            "3. 量词必须与数词搭配使用\n"
            "4. 保留原文的文化特色和表达风格\n"
            "5. 输出只需包含最终翻译结果，不要添加任何额外说明\n\n"
        )

        # 添加上下文信息
        if paragraph_id:
            prompt += f"## 上下文信息:\n当前句子属于段落: {paragraph_id}\n\n"

        # 语法规则
        if top_rules:
            prompt += "## 相关语法规则:\n"
            for i, (rule_dict, score) in enumerate(top_rules[:2], 1):  # 只取前2条规则
                prompt += f"{i}. {PromptConstructor.format_rule(rule_dict)}"
            prompt += "\n"

        # 词典释义
        if dictionary_entries:
            prompt += PromptConstructor.format_dictionary_entries(dictionary_entries)
            prompt += "\n"

        if top_examples:
            prompt += "## 翻译参考示例:\n"
            for i, (example, score) in enumerate(top_examples[:2], 1):
                prompt += f"示例 {i}:\n{PromptConstructor.format_example(example)}"
            prompt += "\n"

        prompt += (
            f"## 待翻译句子:\n"
            f"壮语: {test_item['za']}\n\n"
            "严格遵守上文要求并结合给出示例\n\n"
            "## 所以这句话的翻译结果是（只需汉语无需过多解释）:"
        )
        return prompt