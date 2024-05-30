from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Step 6: Training
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Step 8: Save Model
model.save('model.h5')
