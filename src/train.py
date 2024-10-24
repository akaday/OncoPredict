import optuna
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, clean_data, preprocess_data

def create_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    for i in range(n_layers):
        model.add(Conv2D(trial.suggest_int(f'n_units_l{i}', 32, 128), (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(trial.suggest_int('n_units_dense', 64, 256), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    # Load the dataset
    data = load_data('data/dataset.csv')

    # Clean and preprocess the dataset
    data = clean_data(data)
    X, y = preprocess_data(data)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    model = create_model(trial)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()

    if args.tune:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        print('Best trial:')
        trial = study.best_trial
        print('  Value: {}'.format(trial.value))
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
    else:
        # Load the dataset
        data = load_data('data/dataset.csv')

        # Clean and preprocess the dataset
        data = clean_data(data)
        X, y = preprocess_data(data)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model using the training set and evaluate it using the testing set
        model = create_model(None)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
