/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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
using k = Keras;
using BH.oM.DeepLearning.Layers;
using BH.oM.DeepLearning.Activations;
using BH.oM.DeepLearning;
using BH.oM.DeepLearning.Losses;
using BH.oM.DeepLearning.Optimisers;

namespace BH.Engine.Keras
{
    public static partial class Convert
    {
        /***************************************************/
        /**** Public Methods - General                  ****/
        /***************************************************/

        public static IShape FromKeras(this k.Shape shape)
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

        public static IModule FromKeras(this k.Base module)
        {
            return FromKeras(module as dynamic);
        }


        /***************************************************/
        /**** Public Methods - Enums                    ****/
        /***************************************************/

        public static Reduce FromKerasReduction(string reduction)
        {
            switch (reduction)
            {
                case "auto":
                    return Reduce.Sum;
                case "mean":
                    return Reduce.Mean;
                case "sum":
                    return Reduce.Sum;
                case "none":
                    return Reduce.No;
                default:
                    return Reduce.No;
            }
        }

        /***************************************************/

        public static ImageFormat FromKerasDataFormat(string dataFormat)
        {
            switch(dataFormat)
            {
                case "channels_last":
                    return ImageFormat.ChannelLast;
                default:
                    return ImageFormat.ChannelFirst;
            }
        }

        /***************************************************/

        public static InterpolationMethod FromKerasInterpolationMethod(string interpolation)
        {
            switch(interpolation)
            {
                case "bilinear":
                    return InterpolationMethod.Bilinear;
                case "bicubic":
                    return InterpolationMethod.Bicubic;
                default:
                    return InterpolationMethod.Nearest;
            }
        }


        /***************************************************/
        /**** Public Methods - Activations              ****/
        /***************************************************/

        public static LeakyReLU FromKeras(this k.Layers.LeakyReLU leakyRelu)
        {
            return new LeakyReLU()
            {
                NegativeSlope = (double)leakyRelu.Parameters["alpha"]
            };
        }

        /***************************************************/

        public static LogSigmoid FromKeras(this k.Layers.LogSigmoid logSigmoid)
        {
            return new LogSigmoid();
        }

        /***************************************************/

        public static LogSoftmax FromKeras(this k.Layers.LogSoftmax logSoftmax)
        {
            return new LogSoftmax()
            {
                // This is not possible since tf.nn.log_softmax is a function and does not store parameters
                //Dimension = (int)logSoftmax.Parameters["axis"]  
            };
        }

        /***************************************************/

        public static ReLU FromKeras(this k.Layers.ReLU relu)
        {
            return new ReLU();
        }

        /***************************************************/

        public static Sigmoid FromKeras(this k.Layers.Sigmoid sigmoid)
        {
            return new Sigmoid();
        }

        /***************************************************/

        public static Softmax FromKeras(this k.Layers.Softmax softmax)
        {
            return new Softmax()
            {
                Dimension = (int)softmax.Parameters["axis"]
            };
        }

        /***************************************************/

        public static Tanh FromKeras(this k.Layers.Tanh tanh)
        {
            return new Tanh();
        }


        /***************************************************/
        /**** Public Methods - Losses                   ****/
        /***************************************************/

        public static BCEWithSigmoid FromKeras(this k.Layers.BCEWithLogits bceWithSigmoid)
        {
            return new BCEWithSigmoid()
            {
                Reduce = FromKerasReduction((bceWithSigmoid.Parameters["reduction"] as string)),
            };
        }

        /***************************************************/

        public static BinaryCrossEntropy FromKeras(this k.Layers.BinaryCrossentropy bce)
        {
            return new BinaryCrossEntropy()
            {
                Reduce = FromKerasReduction((bce.Parameters["reduction"] as string)),
            };
        }

        /***************************************************/

        public static CrossEntropy FromKeras(this k.Layers.CategoricalCrossentropy crossEntropy)
        {
            return new CrossEntropy()
            {
                Reduce = FromKerasReduction((crossEntropy.Parameters["reduction"] as string)),
            };
        }

        /***************************************************/

        public static L1 FromKeras(this k.Layers.MeanAbsoluteError mae)
        {
            return new L1()
            {
                Reduce = FromKerasReduction((mae.Parameters["reduction"] as string))
            };
        }

        /***************************************************/

        public static MeanSquareError FromKeras(this k.Layers.MeanSquaredError mse)
        {
            return new MeanSquareError()
            {
                Reduce = FromKerasReduction((mse.Parameters["reduction"] as string))
            };
        }

        /***************************************************/

        public static NegativeLogLikelihood FromKeras(this k.Layers.NegLogLikelihood nll)
        {
            return new NegativeLogLikelihood()
            {
                Reduce = FromKerasReduction((nll.Parameters["reduction"] as string)),
            };
        }


        /***************************************************/
        /**** Public Methods - Operations               ****/
        /***************************************************/

        public static AvgPooling2d FromKeras(this k.Layers.AveragePooling2D avgPool2d, Tuple<int, int, int> inputShape)
        {
            k.Shape kernelSize = (k.Shape)avgPool2d.Parameters["pool_size"];
            k.Shape stride = (k.Shape)avgPool2d.Parameters["strides"];
            return new AvgPooling2d()
            {
                KernelSize = kernelSize.FromKeras() as Shape2d,
                Stride = stride.FromKeras() as Shape2d,
                Padding = new oM.DeepLearning.Shape2d()
                {
                    Dim1 = Query.Padding(inputShape.Item2, kernelSize[0], stride[0]),
                    Dim2 = Query.Padding(inputShape.Item3, kernelSize[1], stride[1])
                }
            };
        }

        /***************************************************/

        public static Convolution2d FromKeras(this k.Layers.Conv2D conv2d, int featuresIn)
        {
            return new Convolution2d()
            {
                FeaturesIn = featuresIn,
                FeaturesOut = (int)conv2d.Parameters["filters"],
                KernelSize = ((k.Shape)conv2d.Parameters["kernel_size"]).FromKeras() as Shape2d,
                Stride = ((k.Shape)conv2d.Parameters["strides"]).FromKeras() as Shape2d,
                Padding = (string)conv2d.Parameters["padding"] == "same" ? new Shape2d { Dim1 = 1, Dim2 = 1 } : new Shape2d { Dim1 = 0, Dim2 = 0 },
                Dilation = ((k.Shape)conv2d.Parameters["dilation_rate"]).FromKeras() as Shape2d,
            };
        }

        /***************************************************/

        public static GRU FromKeras(this k.Layers.GRU gru, int inputSize)
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

        public static LSTM FromKeras(this k.Layers.LSTM gru, int inputSize)
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

        public static MaxPooling2d FromKeras(this k.Layers.MaxPooling2D maxPool2d, IShape inputShape = null)
        {
            return new MaxPooling2d()
            {
                KernelSize = ((k.Shape)maxPool2d.Parameters["pool_size"])?.FromKeras() as Shape2d,
                Stride = ((k.Shape)maxPool2d.Parameters["strides"])?.FromKeras() as Shape2d,
                Padding = (string)maxPool2d.Parameters["padding"] == "same" ? new Shape2d { Dim1 = 1, Dim2 = 1 } : new Shape2d { Dim1 = 0, Dim2 = 0 },
            };
        }

        /***************************************************/

        public static TransposedConvolution2d FromKeras(this k.Layers.Conv2DTranspose transposedConv2d, int featuresIn, Shape2d outSize = null)
        {
            return new TransposedConvolution2d()
            {
                FeaturesIn = featuresIn,
                FeaturesOut = (int)transposedConv2d.Parameters["filters"],
                KernelSize = ((k.Shape)transposedConv2d.Parameters["kernel_size"]).FromKeras() as Shape2d,
                Stride = ((k.Shape)transposedConv2d.Parameters["strides"]).FromKeras() as Shape2d,
                Padding = (string)transposedConv2d.Parameters["padding"] == "same" ? new Shape2d { Dim1 = 1, Dim2 = 1 } : new Shape2d { Dim1 = 0, Dim2 = 0 },
                Dilation = ((k.Shape)transposedConv2d.Parameters["dilation_rate"]).FromKeras() as Shape2d,
                OutputSize = outSize,
            };
        }
        /***************************************************/
        /**** Public Methods - Optimisers               ****/
        /***************************************************/

        public static SGD FromKeras(this k.Optimizers.SGD sgd)
        {
            return new SGD()
            {
                LearningRate = (double)sgd.Parameters["lr"],
                Momentum = (double)sgd.Parameters["momentum"],
                Nesterov = (bool)sgd.Parameters["nesterov"],
                WeightDecay = (double)sgd.Parameters["decay"]
            };
        }

        /***************************************************/

        public static Adam FromKeras(this k.Optimizers.Adam adam)
        {
            return new Adam()
            {
                LearningRate = (double)adam.Parameters["lr"],
                Beta1 = (double)adam.Parameters["beta_1"],
                Beta2 = (double)adam.Parameters["beta_2"],
                WeightDecay = (double)adam.Parameters["decay"]
            };
        }


        /***************************************************/
        /**** Fallback                                  ****/
        /***************************************************/

        private static object FromKeras(object fallback)
        {
            Engine.Reflection.Compute.RecordError($"No Convert method found for {fallback?.GetType()}");
            return null;
        }

        /***************************************************/
    }
}


