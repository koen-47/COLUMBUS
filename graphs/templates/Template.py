class Template:
    class BASE:
        name = "base"
        elements = [{
            "singular": [0.5, 0.5, 1.0],
            "plural": [
                [0.25, 0.5, 0.5],
                [0.75, 0.5, 0.5]
            ]
        }]

    class HIGH:
        name = "high"
        size = 1.0
        elements = [
            [0.5, 1.0]
        ]
        plural_elements = [
            [0.25, 1.0],
            [0.75, 1.0]
        ]

    class REPETITION_TWO:
        name = "repetition_two"
        size = 0.5
        elements = [
            [0.25, 0.5],
            [0.75, 0.5]
        ]

    class REPETITION_FOUR:
        name = "repetition_four"
        size = 0.5
        elements = [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75]
        ]
