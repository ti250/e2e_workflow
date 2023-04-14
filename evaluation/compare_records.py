import copy
import unicodedata

class Statistics(object):
    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def precision(self):
        if self.tp + self.fp == 0:
            return None
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return None
        return self.tp / (self.tp + self.fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        if self.precision is None or self.recall is None:
            return None
        return 2 * (precision * recall) / (precision + recall)

    def __str__(self):
        return "TP: {tp}, FP: {fp}, TN: {tn}, FN:{fn}".format(tp=self.tp,
                                                              fp=self.fp,
                                                              tn=self.tn,
                                                              fn=self.fn)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return Statistics(self.tp + other.tp,
                          self.fp + other.fp,
                          self.tn + other.tn,
                          self.fn + other.fn)


def _get_value(record, field):
    value = None
    try:
        value = record[field]
    except Exception as e:
        pass
    if value == []:
        value = None
    return value


def _fields_compatible(value_a, value_b, is_lax=False):
    field_is_correct = False

    if isinstance(value_a, list) or isinstance(value_a, set):
        if value_b is None:
            return False

        cleaned_bs = [
            unicodedata.normalize("NFKD", val_b).replace(' ', '').lower().strip("(),;.") for val_b in value_b
        ]
        for val_a in value_a:
            normalised_form = unicodedata.normalize("NFKD", val_a).replace(' ', '').lower().strip("(),;.")
            if normalised_form in cleaned_bs:
                field_is_correct = True
                break
            if is_lax:
                for val_b in cleaned_bs:
                    # if val_b in normalised_form or normalised_form in val_b:
                    if val_b in normalised_form:
                        field_is_correct = True
                if field_is_correct:
                    break

    else:
        if isinstance(value_a, str):
            value_a = value_a.replace(' ', '')
        if isinstance(value_b, str):
            value_b = value_b.replace(' ', '')
        if value_a == value_b:
            field_is_correct = True
        if is_lax and isinstance(value_a, str) and isinstance(value_b, str) and (value_a in value_b or value_b in value_a):
            field_is_correct = True

    return field_is_correct


def get_model_list_keypaths(cls):
    model_list_keypaths = []
    for field_name, field in cls.fields.items():
        field_is_list = False
        while hasattr(field, 'field'):
            field = field.field
            field_is_list = True
        if hasattr(field, 'model_class'):
            if field_is_list:
                model_list_keypaths.append(field_name)
            else:
                sub_keypaths = get_model_list_keypaths(field.model_class)
                for keypath in sub_keypaths:
                    model_list_keypaths.append(field_name + '.' + keypath)
    return model_list_keypaths


def compare_records(
    records_a,
    records_b,
    primary_keys={"default": "catalysts.names"},
    ignore_fields=["paper"],
    global_ignore_fields=[],
    verbose=False,
    _parent_name=None,
    lax_fields=[]):
    # records_a are taken to be the "Correct" ones
    # All records passed in should be of the same type
    if len(records_a) == 0 and len(records_b) == 0:
        return None
    elif not (len(records_a) == 0 or len(records_b) == 0) and type(records_a[0]) != type(records_b[0]):
        raise ValueError("Records must be of the same type")
    equivalent_records = {}
    no_equivalent_a = []
    record_type = type(records_a[0]) if len(records_a) else type(records_b[0])
    primary_key = primary_keys[record_type] if record_type in primary_keys.keys() else primary_keys["default"]
    fields = record_type._all_keypaths()
    no_model_list_fields = record_type._all_keypaths(include_model_lists=False)
    if _parent_name is not None:
        ignore_fields = [field.lstrip(f"{_parent_name}.") for field in ignore_fields]

    fields = [field for field in fields if field not in ignore_fields]

    for ignore_field in global_ignore_fields:
        fields = [field for field in fields if ignore_field not in field]

    no_model_list_fields = [field for field in no_model_list_fields if field in fields]
    model_list_fields = get_model_list_keypaths(record_type)

    fields_tp = {}
    for field in fields:
        fields_tp[field] = 0

    fields_fp = copy.copy(fields_tp)
    fields_tn = copy.copy(fields_tp)
    fields_fn = copy.copy(fields_tp)

    for record_a in records_a:
        key = record_a[primary_key]
        found_equivalent = False
        for record_b in records_b:
            if _get_value(record_b, primary_key) == record_a[primary_key]:
                equivalent_records[record_a] = record_b
                records_b.remove(record_b)
                found_equivalent = True
                break
            elif isinstance(key, list) or isinstance(key, set):
                other_key_uncleaned = _get_value(record_b, primary_key)
                other_key = []
                if other_key_uncleaned is not None:
                    for element in other_key_uncleaned:
                        other_key.append(element.replace(' ', ''))
                for element in key:
                    normalised_element = element.replace(' ', '')
                    if normalised_element in other_key:
                        equivalent_records[record_a] = record_b
                        records_b.remove(record_b)
                        found_equivalent = True
                        break
        if not found_equivalent:
            no_equivalent_a.append(record_a)

    for record_a, record_b in equivalent_records.items():
        for field in no_model_list_fields:
            value_a = _get_value(record_a, field)
            value_b = _get_value(record_b, field)
            field_name = field
            is_lax = field_name in lax_fields
            if _parent_name is not None:
                field_name = f"{_parent_name}.{field}"
            if _fields_compatible(value_a, value_b, is_lax):
                if value_a is not None:
                    fields_tp[field] += 1
                    if verbose:
                        print("TP:", field_name, value_a, value_b)
            else:
                if value_b is not None:
                    if verbose:
                        print("FP:", field_name, value_a, value_b)
                    fields_fp[field] += 1
                elif value_a is not None and value_b is None:
                    if verbose:
                        print("FN:", field_name, value_a, value_b)
                    fields_fn[field] += 1
                else:
                    if verbose:
                        print("DID NOT COUNT:", value_a, value_b)
                    pass
        for field in model_list_fields:
            value_a = _get_value(record_a, field)
            value_a = list(value_a) if value_a is not None else []
            value_b = _get_value(record_b, field)
            value_b = list(value_b) if value_b is not None else []
            field_stats = compare_records(value_a, value_b, primary_keys, ignore_fields, global_ignore_fields, verbose, _parent_name=field)

            if field_stats is not None:
                for key, value in field_stats.items():
                    fields_tp[key] += value.tp
                    fields_fp[key] += value.fp
                    fields_fn[key] += value.fn


    for record in no_equivalent_a:
        for field in fields:
            if _get_value(record, field) is not None:
                if verbose:
                    field_name = field
                    if _parent_name is not None:
                        field_name = f"{_parent_name}.{field}"
                    print("FN:", field_name, _get_value(record, field))
                fields_fn[field] += 1

    for record in records_b:
        for field in fields:
            value = _get_value(record, field)
            if value is not None and value != []:
                field_name = field
                if _parent_name is not None:
                    field_name = f"{_parent_name}.{field}"
                if verbose:
                    print("FP:", field_name, value)
                fields_fp[field] += 1

    overall_stats = {}
    for field in fields:
        stats = Statistics(fields_tp[field],
                           fields_fp[field],
                           fields_tn[field],
                           fields_fn[field])
        field_name = field
        if _parent_name is not None:
            field_name = f"{_parent_name}.{field}"
        overall_stats[field_name] = stats

    return overall_stats

