import math

def calculate(expression: str) -> dict:
    """Evaluate a math expression safely."""
    try:
        # Evaluate using only functions/constants from math module
        result = eval(expression, {"__builtins__": None}, math.__dict__)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
