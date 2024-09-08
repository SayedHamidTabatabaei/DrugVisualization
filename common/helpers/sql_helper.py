from decimal import Decimal
from enum import Enum as PyEnum


def convert_value(value):
    if isinstance(value, PyEnum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    return value


def generate_insert_command(entity):
    entity_vars = vars(entity)

    entity_vars.pop('_sa_instance_state', None)

    query = (f"INSERT INTO {entity.__tablename__}({', '.join(entity_vars.keys())}) "
             f"VALUES ({', '.join(['%s' for _ in entity_vars])})")

    values = tuple(convert_value(value) for value in entity_vars.values())

    return query, values


def generate_batch_insert_command(entities):
    if not entities:
        raise ValueError("The entities list cannot be empty")

    entity = entities[0]
    entity_vars = vars(entity)

    entity_vars.pop('_sa_instance_state', None)

    value_placeholders = ', '.join([f'({', '.join(['%s' for _ in entity_vars])})' for _ in entities])

    query = (f"INSERT INTO {entity.__tablename__} ({', '.join(entity_vars.keys())}) "
             f"VALUES {value_placeholders}")

    values = [tuple(convert_value(vars(entity).get(col)) for col in entity_vars.keys()) for entity in entities]
    flat_values = [item for sublist in values for item in sublist]

    return query, flat_values


def generate_batch_insert_check_duplicate_command(entities, update_properties):
    if not entities:
        raise ValueError("The entities list cannot be empty")

    entity = entities[0]
    entity_vars = vars(entity)

    entity_vars.pop('_sa_instance_state', None)

    value_placeholders = f'({', '.join(['%s' for _ in entity_vars])})'

    update_columns = [prop.name if hasattr(prop, 'name') else prop for prop in update_properties]
    update_columns = [col for col in update_columns if col in entity_vars]
    update_clause = ', '.join([f"{col} = VALUES({col})" for col in update_columns])

    query = (f"INSERT INTO {entity.__tablename__} ({', '.join(entity_vars.keys())}) "
             f"VALUES {value_placeholders} "
             f"ON DUPLICATE KEY UPDATE {update_clause}")

    values = [tuple(convert_value(vars(entity).get(col)) for col in entity_vars.keys()) for entity in entities]

    return query, values
