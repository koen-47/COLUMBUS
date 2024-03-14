class Template:
    class BASE:
        name = "base"
        elements = [
            {
                "singular": [0.5, 0.5, 1.0, 0, "center"],
                "plural": [
                    [0.25, 0.5, 0.5, 0, "center"],
                    [0.75, 0.5, 0.5, 0, "center"]
                ]
            }
        ]

    class SingleNode:
        class HIGH:
            name = "high"
            elements = [
                {
                    "singular": [0.5, 1.0, 1.0, 0, "center"],
                    "plural": [
                        [0.25, 1.0, 0.5, 0, "center"],
                        [0.75, 1.0, 0.5, 0, "center"]
                    ]
                }
            ]

        class RIGHT:
            name = "right"
            elements = [
                {
                    "singular": [1.0, 0.5, 1.0, 0, "right"],
                    "plural": [
                        [1.0, 0.25, 1.0, 0, "right"],
                        [1.0, 0.75, 1.0, 0, "right"]
                    ]
                }
            ]

        class REPETITION_FOUR:
            name = "repetition_four"
            elements = [
                {
                    "singular": [0.25, 0.25, 0.5, 0, "center"],
                    "plural": [[0.25, 0.25, 0.5, 0, "center"]]
                },
                {
                    "singular": [0.25, 0.75, 0.5, 0, "center"],
                    "plural": [[0.25, 0.75, 0.5, 0, "center"]]
                },
                {
                    "singular": [0.75, 0.25, 0.5, 0, "center"],
                    "plural": [[0.75, 0.25, 0.5, 0, "center"]]
                },
                {
                    "singular": [0.75, 0.75, 0.5, 0, "center"],
                    "plural": [[0.75, 0.75, 0.5, 0, "center"]]
                }
            ]
