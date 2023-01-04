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

using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicSamples
{
    public class MergeExample
    {
        public static void Run()
        {
            var input = new Input(new Keras.Shape(32, 32));
            //var a = new CuDNNLSTM(32).Set(input);
            var a = new Dense(32, activation: "sigmoid").Set(input);
            //a.Set(input);
            var output = new Dense(1, activation: "sigmoid").Set(a);
            //output.Set(a);

            var model = new Keras.Models.Model(new Input[] { input }, new BaseLayer[] { output });

            //Load train data
            Numpy.NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            var input1 = new Input(new Shape(32, 32, 3));
            var conv1 = new Conv2D(32, (4, 4).ToTuple(), activation: "relu").Set(input1);
            var pool1 = new MaxPooling2D((2, 2).ToTuple()).Set(conv1);
            var flatten1 = new Flatten().Set(pool1);

            var input2 = new Input(new Shape(32, 32, 3));
            var conv2 = new Conv2D(16, (8, 8).ToTuple(), activation: "relu").Set(input2);
            var pool2 = new MaxPooling2D((2, 2).ToTuple()).Set(conv2);
            var flatten2 = new Flatten().Set(pool2);

            var merge = new Concatenate(flatten1, flatten2);


        }
    }
}
