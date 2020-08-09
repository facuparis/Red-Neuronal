# Neural Network
In this project I developed a Neural Network Regression model to predict fluid viscosity from the position vs. time curves provided by the Microviscometer designed by Dr. Morhell and Dr.Pastoriza. The motivation for the network was to solve a problem presented in the measurements when the dynamics of the fluid were altered by the border conditions of the microchannel.

We performed numerous measurements in fluids with different viscosity coefficients and at different temperatures to do a supervised training of the neural network. The input in the model was the position of the meniscus of liquid at each time in the Microviscometer, and the labels were the measurements of viscosity of the same fluids in a Brookfield viscometer.

The neural network developed for the first 5 seconds of the measurement showed a very good generalization, tested in blood plasma measurements, even in those measurements where the dynamics were altered.
