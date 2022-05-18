import json
from collections import defaultdict


def get_dict_content(dict_item: dict, key: str):
    val = dict_item
    for k in key.split("."):
        val = val[k]
    return val


def _pretty_format(
    obj: dict, decimal=3, return_percentage=False, add_percentage_symbol=True
):
    if not isinstance(obj, dict):
        return obj

    pretty = {}
    for key, val in obj.items():
        if isinstance(val, float):
            new_val = val
            if return_percentage:
                new_val = 100 * new_val
            if add_percentage_symbol:
                template = f"{{:.{decimal}f}} %"
            else:
                template = f"{{:.{decimal}f}}"
            new_val = template.format(new_val)
        elif isinstance(val, dict):
            new_val = _pretty_format(
                val,
                decimal=decimal,
                return_percentage=return_percentage,
                add_percentage_symbol=add_percentage_symbol,
            )
        else:
            new_val = val
        pretty[key] = new_val
    return pretty


class PrettyPrintDict(dict):
    def __str__(
        self, indent=2, decimal=3, return_percentage=True, add_percentage_symbol=True
    ):
        pretty_dict = _pretty_format(
            self,
            decimal=decimal,
            return_percentage=return_percentage,
            add_percentage_symbol=add_percentage_symbol,
        )
        return json.dumps(pretty_dict, ensure_ascii=False, indent=indent)

    def to_dict(self) -> dict:
        return dict(self)

    def jsonify(self, **kwargs) -> str:
        return json.dumps(dict(self), ensure_ascii=False, **kwargs)


class PrettyPrintDefaultDict(defaultdict):
    def __str__(
        self, indent=2, decimal=3, return_percentage=True, add_percentage_symbol=True
    ):
        pretty_dict = _pretty_format(
            self,
            decimal=decimal,
            return_percentage=return_percentage,
            add_percentage_symbol=add_percentage_symbol,
        )
        return json.dumps(pretty_dict, ensure_ascii=False, indent=indent)

    def to_dict(self) -> dict:
        return dict(self)

    def jsonify(self, **kwargs) -> str:
        return json.dumps(dict(self), ensure_ascii=False, **kwargs)
