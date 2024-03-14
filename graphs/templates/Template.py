class Template:
    class BASE:
        name = "base"
        elements = [
            {
                "singular": [0.5, 0.5, 1.0],
                "plural": [
                    [0.25, 0.5, 0.5],
                    [0.75, 0.5, 0.5]
                ]
            }
        ]

    class SingleNode:
        class HIGH:
            name = "high"
            elements = [
                {
                    "singular": [0.5, 1.0, 1.0],
                    "plural": [
                        [0.25, 1.0, 0.5],
                        [0.75, 1.0, 0.5]
                    ]
                }
            ]

        class REPETITION_FOUR:
            name = "repetition_four"
            elements = [
                {
                    "singular": [0.25, 0.25, 0.5],
                    "plural": [[0.25, 0.25, 0.5]]
                },
                {
                    "singular": [0.25, 0.75, 0.5],
                    "plural": [[0.25, 0.75, 0.5]]
                },
                {
                    "singular": [0.75, 0.25, 0.5],
                    "plural": [[0.75, 0.25, 0.5]]
                },
                {
                    "singular": [0.75, 0.75, 0.5],
                    "plural": [[0.75, 0.75, 0.5]]
                }
            ]
