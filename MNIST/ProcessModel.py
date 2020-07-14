from keras.models import load_model


# Print useful information about the model (
def print_model_weights(saved_weights):
    # Load model
    model = load_model("%s" % saved_weights)
    # Print a summary of the model (layers and num_parameters)
    model.summary()

    # Print the weights and biases for every layer
    num_layers = len(model.layers)
    for i in range(num_layers):
        # Set next layer and print descriptive info
        curr_layer = model.layers[i]
        print("\nLayer", i, "—", curr_layer.name)

        # Set weights
        # Observe that get_weights() returns weights in [0] and biases in [1]
        weights = curr_layer.get_weights()
        if len(weights) > 0:
            print("\tWeights", i)
            print("\tWeights shape:", weights[0].shape)
            print(weights[0])
            print("\tBiases", i)
            print("\tBiases shape:", weights[1].shape)
            print(weights[1])
        # Skip layers that have no weights (ex: reshaping layers)
        else:
            print("\tEmpty Layer")


# Save the weights of a model into an array
def array_model_weights(saved_weights):
    # Load model and model size
    model = load_model("%s" % saved_weights)
    num_layers = len(model.layers)

    array_weights = [list] * num_layers

    # For every layer, append the layers weights to array_weights
    for i in range(num_layers):
        # Set next layer
        curr_layer = model.layers[i]

        # Set weights
        # Observe that get_weights() returns weights in [0] and biases in [1]
        layer_weights = curr_layer.get_weights()
        array_weights[i] = layer_weights

    return array_weights


# Entry Point
model_name = "final_model.h5"
model_weights = array_model_weights(model_name)

"""
Observe: we use model_weights[j][k] s.t.:
    j is an integer between 0 and num_layers
    k is 0 (for weights) or 1 (for biases)
    
    Examples:
        model_weights[0][0] gives the weights of the first layer in the model
        model_weights[3][1] gives the biases of the fourth layer in the model
"""
