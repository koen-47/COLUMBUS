class Template:
    """
    Class to contain information on the templates. Each template has a name and an elements list. The format of the list
    is as a list of lists, where each inner corresponds to information on an element. The format of this inner element
    is [x_coordinate, y_coordinate, font_size_multiplier]. See the Image Generation paragraph in Section 3.2 for
    more information on this.
    """
    class BASE:
        """
        Basic template for only one element centered in the middle.
        """
        name = "base"
        elements = [[0.5, 0.5, 1.0]]

    class BASE_TWO:
        """
        Basic template for two elements placed side by side.
        """
        name = "base_two"
        elements = [
            [0.25, 0.5, 1.0],
            [0.75, 0.5, 1.0]
        ]

    class BASE_THREE:
        """
        Basic template for three elements placed side by side
        """
        name = "base_three"
        elements = [
            [0.1, 0.5, 0.75],
            [0.5, 0.5, 0.75],
            [0.9, 0.5, 0.75]
        ]

    class ABOVE:
        """
        Template used for the above relational rule.
        """
        name = "above"
        elements = [
            [0.5, 0.65, 0.85],
            [0.5, 0.35, 0.85]
        ]
