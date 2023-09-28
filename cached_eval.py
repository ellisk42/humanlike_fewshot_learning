from utilities import *
from subprocess import STDOUT, check_output
from filelock import FileLock
import tempfile
import os
import func_timeout

"""2 dictionaries are maintained:
1. saved_evaluations: everything that we know is already saved to disk
2. buffered_evaluations: everything that we have computed but not yet saved to disk

Every 100 evaluations, we save the buffer to disk, then clear the buffer.
"""
saved_evaluations = None
buffered_evaluations = {}

def exec_and_return_output(code):
    OUTPUT = []
    exec(code)
    return OUTPUT

def cached_eval(expression, statement=False):
    global saved_evaluations, buffered_evaluations

    fn = "eval.pickle"
    if saved_evaluations is None:
        with FileLock(f'{fn}.lck'):
            if os.path.exists(fn):
                with open(fn, "rb") as handle:
                    saved_evaluations = pickle.load(handle)
            else:
                saved_evaluations = {}
        print("Loaded", len(saved_evaluations), "evaluations from disk")
    
    
    if expression in buffered_evaluations:
        result = buffered_evaluations[expression]
    elif expression in saved_evaluations: 
        result = saved_evaluations[expression]
    else:
        if statement:
            if False: # process sandbox
                # dump code to a temporary file stored on the RAM, /dev/shm
                # this is faster than writing to disk
                with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False, suffix=".py") as f:
                    f.write(expression.encode("utf-8"))
                    temporary_filename = f.name

                # execute the code
                # collect the output
                # delete the temporary file
                try:
                    value = check_output(["python", temporary_filename], stderr=STDOUT, timeout=0.1)
                    value = eval(value)
                except:
                    value = None
                os.remove(temporary_filename)
            else:
                # quasi sanitation
                # make sure that there is no `import` statement
                # make sure that there is no `open` statement
                # make sure that there is no `exec` statement
                # make sure that there is no `eval` statement
                # make sure that there is no `os` statement
                # make sure that there is no `sys` statement
                # make sure that there is no `subprocess` statement
                has_bad_thing = False
                for bad_thing in ["import", "open(", "exec(", "eval(", "os.", "sys.", "subprocess."]:
                    if bad_thing in expression:
                        has_bad_thing = True
                        print(f"bad thing found in generated code, skipping: `{bad_thing}`")
                        break
                if has_bad_thing:
                    value = None
                
                else:
                    modified_expression = expression.replace("print(", "OUTPUT.append(")
                    
                    try:
                        OUTPUT = func_timeout.func_timeout(0.1, exec_and_return_output, args=(modified_expression,))
                        # we expect exactly one output
                        value = eval(str(OUTPUT[-1]))
                        print("successfully executed")
                    except:
                        print("failed to execute")
                        value = None
        else:
            try:
                value = check_output(["python", "-c", expression], stderr=STDOUT, timeout=0.1)
                value = eval(value)
            except:
                value = None
        
        result = [value]
        buffered_evaluations[expression] = result

    if len(buffered_evaluations) > 1000:
        print("dumping buffered evaluations to disk...")
        with FileLock(f'{fn}.lck'):
            # load the saved evaluations again, in case someone else has updated them
            if os.path.exists(fn):
                with open(fn, "rb") as handle:
                    saved_evaluations = pickle.load(handle)
            
            saved_evaluations.update(buffered_evaluations)
            buffered_evaluations = {}
            with open(fn, "wb") as handle:
                pickle.dump(saved_evaluations, handle)

    return result[0]
