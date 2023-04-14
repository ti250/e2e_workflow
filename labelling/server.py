import os
import shutil
import json

from flask import Flask, request, send_from_directory
from pprint import pprint

from cdedatabase import CDEDatabase, JSONCoder
from chemdataextractor.model import ModelType, StringType, InferredProperty, ListType, FloatType, SetType
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

MODELS = []
TYPE_DICTIONARY = {StringType: "string", FloatType: "float"}
CODER = JSONCoder()


@app.route("/api/all_papers")
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def all_paper_names():
    db_root_name = request.args.get("dbName")
    print("\n\n\nDBROOTNAME")
    print(db_root_name)
    try:
        return {"papers": sorted([folder for folder in os.listdir(db_root_name) if os.path.isdir(os.path.join(db_root_name, folder))])}
    except FileNotFoundError:
        return {"papers": []}


def schema_for(model):
    schema = {}
    for field_name, field in model.fields.items():
        if isinstance(field, InferredProperty):
            pass
        elif isinstance(field, ListType) or isinstance(field, SetType):
            content_type = "other"
            if isinstance(field.field, ModelType):
                schema[field_name] = {"type": "modellist", "contentType": field.field.model_name}
            else:
                if type(field.field) in TYPE_DICTIONARY:
                    content_type = TYPE_DICTIONARY[type(field.field)]
                schema[field_name] = {"type": "list", "contentType": content_type}
        elif isinstance(field, ModelType):
            schema[field_name] = {"type": "model", "contentType": field.model_name}
        elif type(field) in TYPE_DICTIONARY.keys():
            schema[field_name] = {"type": TYPE_DICTIONARY[type(field)]}
    return schema


def streamlined_models(models):
    streamlined_models = set()
    for model in models:
        streamlined_models.update(model.flatten(include_inferred=False))
    return streamlined_models


def shallow_jsonified_record(record):
    jsonified = {}
    print("IN SHALLOW JSONZIED RECORD")
    print(record)
    print(type(record))
    pprint(record.serialize())
    for field_name, field in record.fields.items():
        if record[field_name]:
            if isinstance(field, InferredProperty):
                pass
            elif isinstance(field, ModelType):
                jsonified[field_name] = record[field_name]._id
            elif ((isinstance(field, SetType) or isinstance(field, ListType))
                  and isinstance(field.field, ModelType)):
                jsonified[field_name] = [list_item._id for list_item in record[field_name]]
            elif isinstance(field, SetType):
                jsonified[field_name] = list(record[field_name])
            else:
                jsonified[field_name] = record[field_name]
    jsonified["_id"] = record._id
    return jsonified


def deep_jsonified_record(record):
    jsonified_records = [(type(record).__name__, shallow_jsonified_record(record))]
    for field_name, field in record.fields.items():
        if isinstance(field, ModelType):
            if hasattr(record, field_name) and record[field_name] is not None:
                jsonified_records.extend(deep_jsonified_record(record[field_name]))
        elif ((isinstance(field, SetType) or isinstance(field, ListType))
               and isinstance(field.field, ModelType)):
            if hasattr(record, field_name) and record[field_name] is not None:
                for el in record[field_name]:
                    jsonified_records.extend(deep_jsonified_record(el))
    return jsonified_records


@app.route("/api/records")
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def records_for_paper():
    paper_name = request.args.get("paperName")
    db_root_name = request.args.get("dbName")
    db_name = os.path.join(db_root_name, paper_name)
    print(db_name)
    db = CDEDatabase(db_name, coder=CODER)

    all_records = {}
    for model in streamlined_models(MODELS):
        all_records[model.__name__] = []

    for model in MODELS:
        records = db.records(model)
        for record in records:
            print(record.serialize())
            jsonifieds = deep_jsonified_record(record)
            print(jsonifieds)
            for name, jsonified in jsonifieds:
                already_exists = False
                for element in all_records[name]:
                    if element["_id"] == jsonified["_id"]:
                        already_exists = True
                        break
                if not already_exists:
                    all_records[name].append(jsonified)

    return all_records


@app.route("/api/sync", methods=["POST"])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def sync_records():
    given_data = json.loads(request.data)
    given_records = given_data["records"]
    paper_name = given_data["paperName"]
    db_root_name = given_data["dbName"]
    deleted_records = given_data["deletedRecords"]
    print(deleted_records)

    db_name = os.path.join(db_root_name, paper_name)
    print(db_name)
    db = CDEDatabase(db_name, coder=CODER)

    print(given_records)
    all_records = make_records(given_records)
    all_records_flattened = []
    for key, records in all_records.items():
        all_records_flattened.extend(records)

    print("ALL_RECORDS_FLATTENED")
    for record in all_records_flattened:
        print(record.serialize())
    # print(all_records_flattened)
    db.write(all_records_flattened)

    delete_records(deleted_records, db)

    dict_records = {}
    for model in streamlined_models(MODELS):
        dict_records[model.__name__] = []

    for model in MODELS:
        records = all_records[model.__name__]
        for record in records:
            print(record.serialize())
            jsonifieds = deep_jsonified_record(record)
            print(jsonifieds)
            for name, jsonified in jsonifieds:
                already_exists = False
                for element in dict_records[name]:
                    if element["_id"] == jsonified["_id"]:
                        already_exists = True
                        break
                if not already_exists:
                    dict_records[name].append(jsonified)

    return dict_records


def delete_records(deleted_records, db):
    print("DELETING", deleted_records)
    top_level_models = set(MODELS)
    streamlined_models_ = streamlined_models(top_level_models)

    for record_type, deleted_ids in deleted_records.items():
        if not deleted_ids:
            continue
        model_class = None
        for model in streamlined_models_:
            if model.__name__ == record_type:
                model_class = model
                break
        if model_class is None:
            raise ValueError(f"Record with type {record_type} is not supported.")

        print(deleted_ids)
        db.delete(model_class, deleted_ids)
        print("Successfully deleted")


def make_records(given_records):
    top_level_models = set(MODELS)
    streamlined_models_ = streamlined_models(top_level_models)

    all_records = {}
    for model in streamlined_models_:
        all_records[model.__name__] = []

    for record_type, dict_records in given_records.items():
        model_class = None
        for model in streamlined_models_:
            if model.__name__ == record_type:
                model_class = model
                break
        if model_class is None:
            raise ValueError(f"Record with type {record_type} is not supported.")

        schema = schema_for(model_class)

        for dict_record in dict_records:
            add_record_if_needed(dict_record, schema, model_class, record_type, given_records, all_records)

    return all_records


def add_record_if_needed(dict_record, schema, model_class, record_type, given_records, all_records):
    for record in all_records[record_type]:
        if hasattr(record, "tempId") and "tempId" in dict_record.keys():
            if record.tempId == dict_record["tempId"]:
                return record
        elif hasattr(record, "_id") and "_id" in dict_record.keys():
            if record._id == dict_record["_id"]:
                return record

    fields_dict = {}
    for key, value in dict_record.items():
        if key in ["_id", "tempId"]:
            pass
        elif schema[key]["type"] == "model":
            target_model_name = schema[key]["contentType"]
            target_records = all_records[target_model_name]

            child_record = None
            for record in target_records:
                if ((hasattr(record, "_id") and record._id == value)
                    or (hasattr(record, "_id") and record._id == value)):
                    child_record = record

            if child_record is None:
                child_dicts = given_records[target_model_name]
                child_dict = None
                for potential_child in child_dicts:
                    temp_id_condition = ("tempId" in potential_child.keys() and potential_child["tempId"] == value)
                    id_condition = ("_id" in potential_child.keys() and potential_child["_id"] == value)
                    if temp_id_condition or id_condition:
                        child_dict = potential_child
                        break
                if child_dict is None:
                    raise ValueError("Could not find child dict")
                target_model_class = list(filter(lambda model: model.__name__ == target_model_name, streamlined_models(MODELS)))[0]
                target_model_schema = schema_for(target_model_class)
                child_record = add_record_if_needed(child_dict, target_model_schema, target_model_class, target_model_name, given_records, all_records)

            fields_dict[key] = child_record
        elif schema[key]["type"] == "modellist":
            if len(value):
                fields_dict[key] = []
            for el in value:
                target_model_name = schema[key]["contentType"]
                target_records = all_records[target_model_name]

                child_record = None
                for record in target_records:
                    if ((hasattr(record, "_id") and record._id == el)
                        or (hasattr(record, "_id") and record._id == el)):
                        child_record = record

                if child_record is None:
                    child_dicts = given_records[target_model_name]
                    child_dict = None
                    for potential_child in child_dicts:
                        temp_id_condition = ("tempId" in potential_child.keys() and potential_child["tempId"] == el)
                        id_condition = ("_id" in potential_child.keys() and potential_child["_id"] == el)
                        if temp_id_condition or id_condition:
                            child_dict = potential_child
                            break
                    if child_dict is None:
                        raise ValueError("Could not find child dict")
                    target_model_class = list(filter(lambda model: model.__name__ == target_model_name, streamlined_models(MODELS)))[0]
                    target_model_schema = schema_for(target_model_class)
                    child_record = add_record_if_needed(child_dict, target_model_schema, target_model_class, target_model_name, given_records, all_records)

                fields_dict[key].append(child_record)
        else:
            fields_dict[key] = value

    record = model_class(**fields_dict)

    if "_id" in dict_record.keys():
        record._id = int(dict_record["_id"])
    elif "tempId" in dict_record.keys():
        record.tempId = dict_record["tempId"]
    else:
        raise ValueError(f"No id or tempId in record {dict_record}")

    all_records[record_type].append(record)
    return record


@app.route("/api/schema")
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def all_schema():
    top_level_models = set(MODELS)

    streamlined_models_ = streamlined_models(top_level_models)
    # other_models = streamlined_models - top_level_models

    top_level_schema = {model.__name__: schema_for(model) for model in top_level_models}
    all_schema = {model.__name__: schema_for(model) for model in streamlined_models_}
    return {"topLevelSchema": top_level_schema, "allSchema": all_schema}


# @app.route("/api/db_name")
# def db_name():
#     print(DB_ROOT_NAME)
#     return {"dbName": DB_ROOT_NAME}


@app.route("/api/set_db_name", methods=["POST"])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def set_db_name():
    given_data = json.loads(request.data)
    print(given_data)
    db_root_name = given_data["db_root"]
    new_db_name = os.path.join(db_root_name, given_data["new_name"])
    old_db_name = os.path.join(db_root_name, given_data["old_name"])
    print(new_db_name, old_db_name)
    shutil.move(old_db_name, new_db_name)
    return {"status": "success", "papers": [folder for folder in os.listdir(db_root_name) if os.path.isdir(os.path.join(db_root_name, folder))]}


@app.route("/api/add_paper", methods=["POST"])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def add_paper():
    given_data = json.loads(request.data)
    print(given_data)
    db_root_name = given_data["db_root"]
    os.mkdir(os.path.join(db_root_name, given_data["new_name"]))
    return {"status": "success", "papers": [folder for folder in os.listdir(db_root_name) if os.path.isdir(os.path.join(db_root_name, folder))]}


if __name__ == "__main__":
    app.run(debug=True, port=5002)
