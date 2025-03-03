from hkweather import SceneCollection as SC

directory = "data"
collection = SC(directory)
collection.get_scenes()
print(collection.scenes)
collection.process_scenes()
