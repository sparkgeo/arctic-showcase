# def load_scene(scene):
#     return scene

# def yield_chips(schene):
#     for i in [1, 2]:
#         yield i

# def yield_all_chips(scene_list):
#     for scene in scene_list:
#         print(f"Processing scene: {scene}")
#         scene = load_scene(scene)
#         for chip in yield_chips(scene):
#             print(f"Processing chip: {chip} for scene: {scene}")
#             yield chip


# def main():
#     scene_list = ["1", "2", "3"]

#     for chip in yield_all_chips(scene_list):
#         print(f"Yielded chip: {chip}")
