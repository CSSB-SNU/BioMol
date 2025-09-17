from biomol.io.recipe import RecipeBook
from biomol.io.context import ParsingCache
from biomol.io.cookbook import Cooker


def add(b: list[int], c: list[int], d: list[int]) -> list[int]:
    """Add three lists element-wise."""
    return [x + y + z for x, y, z in zip(b, c, d, strict=True)]


data_dict = {
    "B": [1, 2, 3],
    "C": [4, 5, 6],
    "D": [7, 8, 9],
}

parse_cache = ParsingCache()
my_recipe = RecipeBook()
my_recipe.add(
    target={"final": list[int]},
    instruction=add,
    b="A",
    c="C",
    d="D",
)
my_recipe.add(
    target={"A": list[int]},
    instruction=add,
    b="B",
    c="C",
    d="D",
)
my_recipe.add(
    target={"B": list[int]},
    instruction=lambda b: b,
    b="B",
)  # if the instruction is super simple, you can use a lambda function
cookbook = Cooker(parse_cache=parse_cache, recipebook=my_recipe)
# NOTE: why should cooker use external parsingCache ?

# preparing the inputs of whole cooking process
cookbook.prep(data_dict=data_dict, fields=["B", "C", "D"])


cookbook.cook()
A, final = cookbook.serve(output=["A", "final"]).values()
print("A:", A)
print("final:", final)
