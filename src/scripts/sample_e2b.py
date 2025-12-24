from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

load_dotenv()

with Sandbox.create() as sandbox:
    sandbox.run_code("x = 1")
    execution = sandbox.run_code("x+=1; x")
    print(execution.text)  # outputs 2
