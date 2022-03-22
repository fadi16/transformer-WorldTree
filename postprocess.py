from nltk import RegexpTokenizer, PorterStemmer
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words("english")
regexp_tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()


def get_bow_of_fact(fact):
    if fact.isspace() or fact == "":
        raise Exception("fact can't be empty")
    tokenized_fact = regexp_tokenizer.tokenize(fact)
    return set(
        stemmer.stem(word.lower().strip()) for word in tokenized_fact if
        word.lower().strip() not in STOP_WORDS and word != "" and not word.isspace()
    )


def postprocess_explanation(explanation, post_sep=" . "):
    # normalize separators
    explanation = explanation.replace("%%", "$$").replace("&&", "$$").replace("||", "$$")

    facts = [f for f in explanation.split("$$") if f != "" and not f.isspace()]

    postprocessed_facts = [postprocess_fact(fact) for fact in facts if not fact.isspace() and fact != ""]

    # get rid of repeated facts
    postprocessed_facts_no_rep = []

    facts_bows_no_rep = []
    facts_bows = [get_bow_of_fact(fact) for fact in postprocessed_facts]

    for i in range(len(facts_bows)):
        if facts_bows[i] not in facts_bows_no_rep:
            facts_bows_no_rep.append(facts_bows[i])
            postprocessed_facts_no_rep.append(postprocessed_facts[i])

    return post_sep.join(postprocessed_facts_no_rep)


def postprocess_fact(fact):
    return " ".join(regexp_tokenizer.tokenize(fact.lower()))


def regexp_tokenize(exp):
    return regexp_tokenizer.tokenize(exp)
