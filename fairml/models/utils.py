def save_model(model, mean_gen_loss):
    model_json = model.generator.to_json()
    json_path = model.checkpoint_dir + "generator_" + str(mean_gen_loss) + ".json"
    hd5_path = model.checkpoint_dir + "generator_" + str(mean_gen_loss) + ".h5"
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    model.generator.save_weights(hd5_path)
    print("Saved model!")
