
library(keras)
library(tidyverse)
library(tokenizers)
library(tensorflow)

maxlen <- 30

Title <- read_lines("titles.txt") %>% 
	str_to_lower() %>%
	str_c(collapse = "\n") %>%
	tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)

print(sprintf("corpus length: %d", length(Title)))



chars <- Title %>% 
unique() %>% 
sort()

print(sprintf("total chars: %d", length(chars)))

data <- map(
	seq(1, length(Title) - maxlen - 1, by = 3),
	~list(sentence = Title[.x:(.x + maxlen - 1)],
		next_char = Title[.x + maxlen])
)

data <- transpose(data)



x <- array(0, dim = c(length(data$sentence), maxlen, length(chars)))
y <- array(0, dim = c(length(data$sentence), length(chars)))

for (i in 1:length(data$sentence)){

	x[i,,] <- sapply(chars, function(x){
		as.integer(x == data$sentence[[i]])
		})

	y[i,] <- as.integer(chars == data$next_char[[i]])

}


model <- keras_model_sequential()

model %>%
	layer_lstm(128, input_shape = c(maxlen, length(chars))) %>%
	layer_dense(length(chars)) %>%
	layer_activation("softmax")

optimizer <- optimizer_rmsprop(lr = 0.01)

model %>% compile(
	loss = "categorical_crossentropy",
	optimizer = optimizer
)


sample_mod <- function(preds, temperature = 1){
	preds <- log(preds)/temperature
	exp_preds <- exp(preds)
	preds <- exp_preds/sum(exp(preds))

	rmultinom(1, 1, preds) %>%
		as.integer() %>%
		which.max()
}

on_epoch_end <- function(epoch, logs){

	cat(sprintf("epoch: %02d ---------------\n\n", epoch))

	for(diversity in c(0.2, 0.5, 1, 1.2)){

		cat(sprintf("epoch: %02d ---------------\n\n", epoch))

		start_index <- sample(1:(length(Title) - maxlen), size = 1)
		sentence <- Title[start_index:(start_index + maxlen - 1)]
		generated <- ""

		for(i in 1:400){

			x <- sapply(chars, function(x){
				as.integer(x == sentence)
				})
			x <- array_reshape(x, c(1, dim(x)))

			preds <- predict(model, x)
			next_index <- sample_mod(preds, diversity)
			next_char <- chars[next_index]

			generated <- str_c(generated, next_char, collapse = "")
			sentence <- c(sentence[-1], next_char)

		}

		cat(generated)
		cat("\n\n")

	}
}

print_callback <- callback_lambda(on_epoch_end = on_epoch_end)

model %>% fit(
	x, y,
	batch_size = 128,
	epochs = 20,
	callbacks = print_callback
)
