/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

using Keras.Datasets;
using System;
using Numpy;
using Keras.Models;
using Keras.Layers;
using Keras.PreProcessing.sequence;
using Keras.PreProcessing.Text;
using System.Linq;
namespace TextExamples
{
    public class SentimentClassification
    {
        public static void Run()
        {
            //Load IMDb dataset
            var ((x_train, y_train), (x_test, y_test)) = IMDB.LoadData();

            var X = np.concatenate(new NDarray[] { x_train, x_test }, axis: 0);
            var Y = np.concatenate(new NDarray[] { y_train, y_test }, axis: 0);

            Console.WriteLine("Shape of X: " + X.shape);
            Console.WriteLine("Shape of Y: " + Y.shape);

            //We can get an idea of the total number of unique words in the dataset.
            Console.WriteLine("Number of words: ");
            var hstack = np.hstack(new NDarray[] { X });
            //var unique = hstack.unique();
            //Console.WriteLine(np.unique(np.hstack(new NDarray[] { X })).Item1);

            // Load the dataset but only keep the top n words, zero the rest
            int top_words = 5000;
            ((x_train, y_train), (x_test, y_test)) = IMDB.LoadData(num_words: top_words);

            int max_words = 500;
            x_train = SequenceUtil.PadSequences(x_train, maxlen: max_words);
            x_test = SequenceUtil.PadSequences(x_test, maxlen: max_words);

            //Create model
            Sequential model = new Sequential();
            model.Add(new Embedding(top_words, 32, input_length: max_words));
            model.Add(new Conv1D(filters: 32, kernel_size: 3, padding: "same", activation: "relu"));
            model.Add(new MaxPooling1D(pool_size: 2));
            model.Add(new Flatten());
            model.Add(new Dense(250, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(loss: "binary_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });
            model.Summary();

            // Fit the model
            model.Fit(x_train, y_train, validation_data: new NDarray[] { x_test, y_test }, epochs: 10, batch_size: 128, verbose: 2);
            // Final evaluation of the model
            var scores = model.Evaluate(x_test, y_test, verbose: 0);
            Console.WriteLine("Accuracy: " + (scores[1] * 100));

            model.Save("model.h5");
            model.SaveTensorflowJSFormat("./");
        }

        public static void Predict(string text)
        {
            var model = Sequential.LoadModel("model.h5");
            string result = "";

            var indexes = IMDB.GetWordIndex();

            string[] words = TextUtil.TextToWordSequence(text);
            float[] tokens = words.Select(i => ((float)indexes[i])).ToArray();

            NDarray x = np.array(tokens);
            x = x.reshape(1, x.shape[0]);
            x = SequenceUtil.PadSequences(x, maxlen: 500);
            var y = model.Predict(x);
            var binary = Math.Round(y[0].asscalar<float>());
            result = binary == 0 ? "Negative" : "Positive";
            Console.WriteLine("Sentiment for \"{0}\": {1}", text, result);
        }
    }
}
