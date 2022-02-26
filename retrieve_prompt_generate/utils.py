import nltk
from nltk import RegexpTokenizer

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class Utils:
    def __init__(self):
        # tokenizer to only allow alpha-neumeric characters
        self.tokenizer = RegexpTokenizer(r'\w+')

    def init_explanation_bank_lemmatizer(self):
        lemmatization_file = open("./retrieve_prompt_generate/lemmatization-en.txt")
        self.lemmas = {}
        # saving lemmas
        for line in lemmatization_file:
            self.lemmas[line.split("\t")[1].lower().replace("\n", "")] = line.split("\t")[0].lower()
        return self.lemmas

    def explanation_bank_lemmatize(self, string: str):
        if self.lemmas == None:
            self.init_explanation_bank_lemmatizer()
        temp = []
        for word in string.split(" "):
            if word.lower() in self.lemmas:
                temp.append(self.lemmas[word.lower()])
            else:
                temp.append(word.lower())
        return " ".join(temp)

    # def preprocess_fact(self, fact):
    #     fact_tokens = self.tokenizer.tokenize(fact)
    #     lemmatized_fact_tokens = [self.explanation_bank_lemmatize(token.lower()) for token in fact_tokens]
    #     if len(lemmatized_fact_tokens) > 1:
    #         return " ".join(lemmatized_fact_tokens)
    #
    # def preprocess_question(self, question, remove_stopwords=False):
    #     question_tokens = nltk.word_tokenize(question)
    #     lemmatized_question_tokens = [self.explanation_bank_lemmatize(token.lower()) for token in question_tokens
    #                                   if not token.lower() in stopwords.words("english") or not remove_stopwords]
    #     return " ".join(lemmatized_question_tokens)

    def preprocess(self, sent):
        tokens = self.tokenizer.tokenize(sent)
        lemmatized_tokens = [self.explanation_bank_lemmatize(token.lower()) for token in tokens if not token.lower() in stopwords.words("english")]
        return " ".join(lemmatized_tokens)