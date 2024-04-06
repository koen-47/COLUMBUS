class Template:
    class BASE:
        name = "base"
        elements = [[0.5, 0.5, 1.0]]

    class BASE_TWO:
        name = "base_two"
        elements = [
            [0.25, 0.5, 1.0],
            [0.75, 0.5, 1.0]
        ]

    class BASE_THREE:
        name = "base_three"
        elements = [
            [0.1, 0.5, 0.75],
            [0.5, 0.5, 0.75],
            [0.9, 0.5, 0.75]
        ]

# class Template:
#     class BASE_VERTICAL:
#         name = "base_vertical"
#         elements = [
#             {
#                 "singular": [0.5, 0.5, 1.0],
#                 "plural": [
#                     [0.5, 0.25, 1.],
#                     [0.5, 0.75, 1.]
#                 ]
#             }
#         ]
#
#     class BASE_HORIZONTAL:
#         name = "base_horizontal"
#         elements = [
#             {
#                 "singular": [0.5, 0.5, 1.0],
#                 "plural": [
#                     [0.25, 0.5, 0.5],
#                     [0.75, 0.5, 0.5]
#                 ]
#             }
#         ]

    # class SingleNode:
    #     class HIGH:
    #         name = "high"
    #         elements = [
    #             {
    #                 "singular": [0.5, 1.0, 1.0],
    #                 "plural": [
    #                     [0.25, 1.0, 0.5, 0],
    #                     [0.75, 1.0, 0.5, 0]
    #                 ]
    #             }
    #         ]
    #
    #     class RIGHT:
    #         name = "right"
    #         elements = [
    #             {
    #                 "singular": [1.0, 0.5, 1.0],
    #                 "plural": [
    #                     [1.0, 0.25, 1.0, 0],
    #                     [1.0, 0.75, 1.0, 0]
    #                 ]
    #             }
    #         ]
    #
    #     class LEFT:
    #         name = "left"
    #         elements = [
    #             {
    #                 "singular": [0., 0.5, 1.0],
    #                 "plural": [
    #                     [0., 0.25, 1.0, 0],
    #                     [0., 0.75, 1.0, 0]
    #                 ]
    #             }
    #         ]
    #
    #     class LOW:
    #         name = "low"
    #         elements = [
    #             {
    #                 "singular": [0.5, 0., 1.0],
    #                 "plural": [
    #                     [0.25, 0., 0.5],
    #                     [0.75, 0., 0.5]
    #                 ]
    #             }
    #         ]
    #
    #     class REPETITION_FOUR:
    #         name = "repetition_four"
    #         elements = [
    #             {
    #                 "singular": [0.25, 0.25, 0.5],
    #                 "plural": [[0.25, 0.25, 0.5]]
    #             },
    #             {
    #                 "singular": [0.25, 0.75, 0.5],
    #                 "plural": [[0.25, 0.75, 0.5]]
    #             },
    #             {
    #                 "singular": [0.75, 0.25, 0.5],
    #                 "plural": [[0.75, 0.25, 0.5]]
    #             },
    #             {
    #                 "singular": [0.75, 0.75, 0.5],
    #                 "plural": [[0.75, 0.75, 0.5]]
    #             }
    #         ]
