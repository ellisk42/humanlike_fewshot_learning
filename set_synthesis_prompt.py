def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if it is not a small object, and not a green object.
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if it is not a small object, and not a green object.
    #START
    return (not this_size == 1) and (not this_color == "green")
#DONE

def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if it is bigger than every other object
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if it is bigger than every other object
    #START
    return all( this_size > other_object[2] for other_object in other_objects )
#DONE

def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if it is one of the largest
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if it is one of the largest
    #START
    return all( this_size >= other_object[2] for all_example_object in all_example_objects )
#DONE


def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if it is smaller than every yellow object
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if it is smaller than every yellow object
    #START
    return all( this_size < other_object[2] for other_object in other_objects if other_object[1] == "yellow" )
#DONE

def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if there is another object with the same color
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if there is another object with the same color
    #START
    return any( this_color == other_object[1] for other_object in other_objects )
#DONE

def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if it has a unique combination of color and shape
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if it has a unique combination of color and shape
    #START
    return all( this_shape != other_object[0] or this_color != other_object[1] for other_object in other_objects )
#DONE

def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if it has the same color as the majority of objects
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if it has the same color as the majority of objects
    #START
    majority_color = max(["yellow", "green", "blue"], key=lambda color: sum(1 for obj in all_example_objects if obj[1] == color))
    return this_color == majority_color
#DONE

def check_object(this_object, other_objects):
    """
    this_object: a tuple of (shape, color, size)
    other_objects: a list of tuples of (shape, color, size)

    returns: True if `this_object` is positive according to the following rule:
        Something is positive if there are at least two other objects with the same shape
    """
    # shape: a string, either "circle", "rectangle", or "triangle" 
    # color: a string, either "yellow", "green", or "blue"
    # size: an int, either 1 (small), 2 (medium), or 3 (large)
    this_shape, this_color, this_size = this_object
    
    # `this_object` is not a part of `other_objects`
    # to get all of the examples, you can use `all_example_objects`, defined as `other_objects + [this_object]`
    # be careful as to whether you should be using `all_example_objects` or `other_objects` in your code
    all_example_objects = other_objects + [this_object]
    
    # Something is positive if there are at least two other objects with the same shape
    #START
    return sum(1 for other_object in other_objects if other_object[0] == this_shape) >= 2
#DONE

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
    
    # %s
    #START