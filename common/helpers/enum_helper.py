def get_enum_from_string(enum_class, string_value):
    string_value = string_value.lower().replace(" ", "_").capitalize()
    try:
        return enum_class[string_value]
    except KeyError:
        raise ValueError(f"No matching enum found for: {string_value}")