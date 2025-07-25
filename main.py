from fc import *
from utils import *
import json

USE_SBERT = True
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if __name__ == "__maim__":
    # initialize
    grammar_book_path = "file_to_use/grammar_book.json"
    dictionary_path = "file_to_use/dictionary_za2zh.jsonl"
    parallel_corpus_path = "file_to_use/parallel_corpus.json"

    sample_path = "results/sample_submission.csv"
    results_path = "results/translation_prompt.json"
    test_data_path = "file_to_use/test_data_hard.json"
    acquire_json_path="file_to_use/acquire_json.json"

    book_retriever = EnhancedGrammarBookRetriever(grammar_book_path)
    dictionary = EnhancedDictionary(dictionary_path)
    parallel_retriever = ParallelCorpusRetriever(parallel_corpus_path)

    if book_retriever.model_loaded:
        parallel_retriever.set_semantic_model(book_retriever.semantic_model)
        if parallel_retriever.corpus:
            try:
                corpus_texts = [item['za'] for item in parallel_retriever.corpus]
                parallel_retriever.semantic_embeddings = book_retriever.semantic_model.encode(corpus_texts)

            except Exception as e:
                print("Error")

    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

    except Exception as e:
        raise EOFError

    try:
        with open(acquire_json_path,'r',encoding='utf-8') as f:
            acquire_json=json.load(f)
    except Exception as e:
        create_acquire_json(acquire_json_path)
        with open(acquire_json_path,'r',encoding='utf-8') as f:
            acquire_json=json.load(f)


    translation_results = []

    for i, test_item in enumerate(test_data):
        top_rules = book_retriever.find_top_rules(test_item['za'])

        dictionary_entries = []
        words = tokenize_sentence(test_item['za'])
        for word in words:
            entries = dictionary.lookup(word)
            if entries:
                dictionary_entries.append(entries[0])

        top_examples = parallel_retriever.find_top_examples(test_item['za'], top_k=3)

        prompt = PromptConstructor.construct_prompt(
            test_item,
            top_rules,
            dictionary_entries,
            top_examples
        )

        try:
            res = acquire(prompt,acquire_json)

        except Exception as e:
            res = "Fail to translate"

        result_details = {
            "index": i,
            "za_sentence": test_item['za'],
            "zh_translation": res,
            "prompt": prompt,
        }
        translation_results.append(result_details)

        modify_csv(sample_path, i + 1, res)

        if (i + 1) % 10 == 0:
            save_translation_results(translation_results, results_path)

    save_translation_results(translation_results, results_path)
