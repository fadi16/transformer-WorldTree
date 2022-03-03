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

def get_training_exp_role_from_wtv2_exp_role(role):
    # word tree has an extra role: background -> we treat it as grounding
    # word tree has some weird roles: "ROLE", "NER"
    if role == BACKGROUND:
        return GROUNDING
    else:
        return role
