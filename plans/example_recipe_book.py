from biomol.io.context import ParsingContext
from biomol.io.cookbook import CookBook


def add(b:list[int], c:list[int], d:list[int]) -> list[int]:
    """Add three lists element-wise."""
    return [x+y+z for x,y,z in zip(b,c,d,strict=True)]

data_dict = {
    "B": [1,2,3],
    "C": [4,5,6],
    "D": [7,8,9],
}

parse_context = ParsingContext()
cookbook = CookBook(parse_context=parse_context)
cookbook.prep(
    data_dict=data_dict,
    fields=["B","C","D"],
)
cookbook.add_recipe(
    output="final",
    instructions=add,
    inputs=["A","C","D"],
    params=None,
)
cookbook.add_recipe(
    output="A",
    instructions=add,
    inputs=["B","C","D"],
    params=None,
)

cookbook.cook()
A, final = cookbook.serve(output=["A", "final"]).values()
print("A:", A)
print("final:", final)



