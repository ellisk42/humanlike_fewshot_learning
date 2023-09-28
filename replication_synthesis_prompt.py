Please write a python function to check if an object obeys a logical rule. The logical rule talks about the following features:

shape: a string, either "circle", "rectangle", or "triangle" 
color: a string, either "yellow", "green", or "blue"
size: an int, either 1 (small), 2 (medium), or 3 (large)

The python function should be called `check_object`, and inputs:

`this_object`: a tuple of (shape, color, size)
`other_objects`: a list of tuples of (shape, color, size)

The logical rule should check if `this_object` has a certain relationship with `other_objects`. Collectively, `[this_object]+other_objects` correspond to all of the objects, so if the rule references the whole examples, it is talking about that structure.

The logical rule is: %s

Please start your response by writing the following code, and then complete the function body so that it returns `True` if and only if the logical rule above holds.

```
def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        %s
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]

    # return True if and only if:
    # %s
```
