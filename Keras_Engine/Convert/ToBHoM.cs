/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2019, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;
using k = Keras;
using BH.oM.DeepLearning.Layers;
using BH.oM.DeepLearning.Activations;
using System.Linq;
using BH.oM.DeepLearning;

namespace BH.Engine.Keras
{
    public static partial class Convert
    {
        /***************************************************/
        /**** Public Methods - General                  ****/
        /***************************************************/

        public static IShape ToBHoM(this k.Shape shape)
        {
            switch (shape.Dimensions.Length)
            {
                case (2):
                    return new Shape2d() { Dim1 = shape.Dimensions[0], Dim2 = shape.Dimensions[1] };
                case (3):
                    return new Shape3d() { Dim1 = shape.Dimensions[0], Dim2 = shape.Dimensions[1], Dim3 = shape.Dimensions[2] };
                default:
                    return null;
            }
        }


        /***************************************************/
        /**** Public Methods - Activations              ****/
        /***************************************************/

        public static LeakyReLU ToBHoM(this k.Layers.LeakyReLU leakyRelu)
        {
            return new LeakyReLU()
            {
                NegativeSlope = (double)leakyRelu.Parameters["alpha"]
            };
        }

        /***************************************************/

        public static ReLU ToBHoM(this k.Layers.ReLU leakyRelu)
        {
            return new ReLU();
        }

        /***************************************************/

        //public static Sigmoid ToBHoM(this k.Layers.Sigmoid leakyRelu)
        //{
        //    return new Sigmoid();
        //}

        /***************************************************/

        public static Softmax ToBHoM(this k.Layers.Softmax softmax)
        {
            return new Softmax()
            {
                Dimension = (int)softmax.Parameters["axis"]
            };
        }

        /***************************************************/

        //public static Tanh ToBHoM(this k.Layers.Tanh tanh)
        //{
        //    return new Tanh();
        //}


        /***************************************************/
        /**** Public Methods - Operations               ****/
        /***************************************************/

        public static AvgPooling2d ToBHoM(this k.Layers.AveragePooling2D avgPool2d, Tuple<int, int, int> inputShape)
        {
            k.Shape kernelSize = (k.Shape)avgPool2d.Parameters["pool_size"];
            k.Shape stride = (k.Shape)avgPool2d.Parameters["strides"];
            return new AvgPooling2d()
            {
                KernelSize = kernelSize.ToBHoM() as Shape2d,
                Stride = stride.ToBHoM() as Shape2d,
                Padding = new oM.DeepLearning.Shape2d()
                {
                    Dim1 = Query.Padding(inputShape.Item2, kernelSize[0], stride[0]),
                    Dim2 = Query.Padding(inputShape.Item3, kernelSize[1], stride[1])
                }
            };
        }

        /***************************************************/

        public static Convolution2d ToBHoM(this k.Layers.Conv2D conv2d, int featuresIn)
        {
            return new Convolution2d()
            {
                FeaturesIn = featuresIn,
                FeaturesOut = (int)conv2d.Parameters["filters"],
                KernelSize = ((k.Shape)conv2d.Parameters["kernel_size"]).ToBHoM() as Shape2d,
                Stride = ((k.Shape)conv2d.Parameters["strides"]).ToBHoM() as Shape2d,
                Padding = (string)conv2d.Parameters["padding"] == "same" ? new Shape2d { Dim1 = 1, Dim2 = 1 } : new Shape2d { Dim1 = 0, Dim2 = 0 },
                Dilation = ((k.Shape)conv2d.Parameters["dilation_rate"]).ToBHoM() as Shape2d,
            };
        }

        /***************************************************/

        public static GRU ToBHoM(this k.Layers.GRU gru, int inputSize)
        {
            return new GRU()
            {
                InputSize = inputSize,
                HiddenSize = (int)gru.Parameters["units"],
                NumberOfLayers = 1,
                BatchFirst = true,
                Dropout = (double)gru.Parameters["dropout"],
                Bidirectional = (bool)gru.Parameters["go_backwards"],
            };
        }

        /***************************************************/

        public static LSTM ToBHoM(this k.Layers.LSTM gru, int inputSize)
        {
            return new LSTM()
            {
                InputSize = inputSize,
                HiddenSize = (int)gru.Parameters["units"],
                NumberOfLayers = 1,
                BatchFirst = true,
                Dropout = (double)gru.Parameters["dropout"],
                Bidirectional = (bool)gru.Parameters["go_backwards"],
            };
        }

        /***************************************************/

        public static MaxPooling2d ToBHoM(this k.Layers.MaxPooling2D maxPool2d, IShape inputShape = null)
        {
            return new MaxPooling2d()
            {
                KernelSize = ((k.Shape)maxPool2d.Parameters["pool_size"])?.ToBHoM() as Shape2d,
                Stride = ((k.Shape)maxPool2d.Parameters["strides"])?.ToBHoM() as Shape2d,
                Padding = (string)maxPool2d.Parameters["padding"] == "same" ? new Shape2d { Dim1 = 1, Dim2 = 1 } : new Shape2d { Dim1 = 0, Dim2 = 0 },
            };
        }

        /***************************************************/

        public static TransposedConvolution2d ToBHoM(this k.Layers.Conv2DTranspose transposedConv2d, int featuresIn, Shape2d outSize = null)
        {
            return new TransposedConvolution2d()
            {
                FeaturesIn = featuresIn,
                FeaturesOut = (int)transposedConv2d.Parameters["filters"],
                KernelSize = ((k.Shape)transposedConv2d.Parameters["kernel_size"]).ToBHoM() as Shape2d,
                Stride = ((k.Shape)transposedConv2d.Parameters["strides"]).ToBHoM() as Shape2d,
                Padding = (string)transposedConv2d.Parameters["padding"] == "same" ? new Shape2d { Dim1 = 1, Dim2 = 1 } : new Shape2d { Dim1 = 0, Dim2 = 0 },
                Dilation = ((k.Shape)transposedConv2d.Parameters["dilation_rate"]).ToBHoM() as Shape2d,
                OutputSize = outSize,
            };
        }

        /***************************************************/

    }
}
