explanatory_role_to_sep = {
    "CENTRAL": " && ",
    "GROUNDING": " $$ ",
    "BACKGROUND": " $$ ",
    "LEXGLUE": " %% "
}

CENTRAL = "CENTRAL"
GROUNDING = "GROUNDING"
BACKGROUND = "BACKGROUND"
LEXGLUE = "LEXGLUE"

allowed_explanatory_roles = {CENTRAL, GROUNDING, BACKGROUND, LEXGLUE}

CENTRAL_RETRIEVED = "CENTRAL_RETRIEVED"
GROUNDING_RETRIEVED = "GROUNDING_RETRIEVED"
LEXGLUE_RETRIEVED = "LEXGLUE_RETRIEVED"


MAIN_FACTS_SEP = "$$"
EXPLANATORY_ROLES_FACTS_SEP = "||"
CENTRAL_FACTS_SEP = "&&"
GROUNDING_FACTS_SEP = "$$"
BACKGROUND_FACTS_SEP = "$$"
LEXGLUE_FACTS_SEP = "%%"
QUESTION_AND_RETRIEVED_FACTS_SEP = "@@"
RETRIEVED_FACTS_SEP = "££"
START_SEP = "<start>"
END_SEP = "<end>"

EXPLANATORY_ROLES_FACTS_SEP_RE = "\|\|"

ALL_FACTS_SEPARATORS = {
    MAIN_FACTS_SEP, EXPLANATORY_ROLES_FACTS_SEP, CENTRAL_FACTS_SEP,
    GROUNDING_FACTS_SEP, BACKGROUND_FACTS_SEP, LEXGLUE_FACTS_SEP,
    QUESTION_AND_RETRIEVED_FACTS_SEP, RETRIEVED_FACTS_SEP, END_SEP
}

ALL_FACTS_SEPARATORS_RE = {
    EXPLANATORY_ROLES_FACTS_SEP_RE, CENTRAL_FACTS_SEP,
    GROUNDING_FACTS_SEP, BACKGROUND_FACTS_SEP, LEXGLUE_FACTS_SEP,
    QUESTION_AND_RETRIEVED_FACTS_SEP, RETRIEVED_FACTS_SEP, END_SEP
}

def get_training_exp_role_from_wtv2_exp_role(role):
    # word tree has an extra role: background -> we treat it as grounding
    # word tree has some weird roles: "ROLE", "NER"
    if role == BACKGROUND:
        return GROUNDING
    else:
        return role
