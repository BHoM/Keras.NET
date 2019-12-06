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
using k = Keras;
using BH.oM.DeepLearning;
using BH.oM.DeepLearning.Layers;
using BH.oM.DeepLearning.Activations;
using BH.oM.DeepLearning.Losses;
using System.Linq;
using System.Collections.Generic;

namespace BH.Engine.Keras
{
    public static partial class Convert
    {
        /***************************************************/
        /**** Public Methods - Interfaces               ****/
        /***************************************************/

        public static k.Base IToKeras(this IModule module)
        {
            return ToKeras(module as dynamic);
        }

        /***************************************************/
        /**** Public Methods - Enums                    ****/
        /***************************************************/

        public static string ToKeras(this Reduce reduction)
        {
            switch(reduction)
            {
                case Reduce.Mean:
                    return "mean";
                case Reduce.Sum:
                    return "sum";
                case Reduce.No:
                    return "none";
                default:
                    return "none";
            }
        }

        /***************************************************/
        /**** Public Methods - Shape                    ****/
        /***************************************************/

        public static k.Shape IToKeras(this IShape shape)
        {
            return ToKeras(shape as dynamic);
        }

        /***************************************************/

        public static k.Shape ToKeras(this Shape2d shape)
        {
            return new k.Shape(shape.Dim1, shape.Dim2);
        }

        /***************************************************/

        public static k.Shape ToKeras(this Shape3d shape)
        {
            return new k.Shape(shape.Dim1, shape.Dim2, shape.Dim3);
        }

        /***************************************************/
        /**** Public Methods - Models	                ****/
        /***************************************************/

        public static k.Models.Sequential ToKeras(this oM.DeepLearning.Models.Sequential sequential)
        {
            k.Models.Sequential kerasModel = new k.Models.Sequential();

            List<k.Layers.BaseLoss> losses = new List<k.Layers.BaseLoss>();
            foreach(IModule module in sequential.Modules)
            {
                k.Base layer = module.IToKeras();
                switch(layer)
                {
                    case k.Layers.BaseLayer operation:
                        kerasModel.Add(operation);
                        continue;
                    case k.Layers.BaseLoss loss:
                        losses.Add(loss);
                        continue;
                    default:
                        continue;
                }
            }

            return kerasModel;
        }

        /***************************************************/

        public static k.Models.Model ToKeras(this oM.DeepLearning.Models.Graph graph)
        {
            throw new NotImplementedException("The keras.model.Model interface has not been implemented yet.");
        }


        /***************************************************/
        /**** Public Methods - Activations              ****/
        /***************************************************/

        public static k.Layers.LeakyReLU ToKeras(this LeakyReLU leakyRelu)
        {
            k.Layers.LeakyReLU kerasRelu = new k.Layers.LeakyReLU();
            kerasRelu.Parameters["alpha"] = leakyRelu.NegativeSlope;
            return kerasRelu;
        }

        /***************************************************/

        public static k.Layers.LogSigmoid ToKeras(this LogSigmoid logSigmoid)
        {
            return new k.Layers.LogSigmoid();
        }

        /***************************************************/

        public static k.Layers.LogSoftmax ToKeras(this LogSoftmax logSoftmax)
        {
            return new k.Layers.LogSoftmax(logSoftmax.Dimension);
        }

        /***************************************************/

        public static k.Layers.ReLU ToKeras(this ReLU relu)
        {
            return new k.Layers.ReLU();
        }

        /***************************************************/

        public static k.Layers.Sigmoid ToKeras(this Sigmoid sigmoid)
        {
            return new k.Layers.Sigmoid();
        }

        /***************************************************/

        public static k.Layers.Softmax ToKeras(this Softmax softmax)
        {
            return new k.Layers.Softmax(softmax.Dimension);
        }

        /***************************************************/

        public static k.Layers.Tanh ToKeras(this Tanh tanh)
        {
            return new k.Layers.Tanh();
        }


        /***************************************************/
        /**** Public Methods - Losses                   ****/
        /***************************************************/

        public static k.Layers.BCEWithLogits ToKeras(this BCEWithSigmoid bceWithSigmoid)
        {
            return new k.Layers.BCEWithLogits(bceWithSigmoid.Reduce.ToKeras());
        }
        
        /***************************************************/

        public static k.Layers.BinaryCrossentropy ToKeras(this BinaryCrossEntropy bce)
        {
            return new k.Layers.BinaryCrossentropy(bce.Reduce.ToKeras());
        }

        /***************************************************/

        public static k.Layers.CategoricalCrossentropy ToKeras(this CrossEntropy crossEntropy)
        {
            return new k.Layers.CategoricalCrossentropy(crossEntropy.Reduce.ToKeras());
        }

        /***************************************************/

        public static k.Layers.MeanAbsoluteError ToKeras(this L1 mae)
        {
            return new k.Layers.MeanAbsoluteError(mae.Reduce.ToKeras());
        }

        /***************************************************/

        public static k.Layers.MeanSquaredError ToKeras(this MeanSquareError mse)
        {
            return new k.Layers.MeanSquaredError(mse.Reduce.ToKeras());
        }

        /***************************************************/

        public static k.Layers.NegLogLikelihood ToKeras(this NegativeLogLikelihood nll)
        {
            return new k.Layers.NegLogLikelihood(nll.Reduce.ToKeras());
        }


        /***************************************************/
        /**** Public Methods - Layers                   ****/
        /***************************************************/

        public static k.Layers.AveragePooling2D ToKeras(this AvgPooling2d avgPool2d)
        {
            return new k.Layers.AveragePooling2D(
                pool_size: avgPool2d.KernelSize.ToTuple(),
                strides: avgPool2d.Stride.ToTuple(),
                padding: avgPool2d.Padding.Dim1 == 0 && avgPool2d.Padding.Dim2 == 0 ? "valid" : "same",
                data_format: "channel_first");
        }

        /***************************************************/

        public static k.Layers.Conv2D ToKeras(this Convolution2d conv2d)
        {
            return new k.Layers.Conv2D(
            filters: conv2d.FeaturesOut,
            kernel_size: conv2d.KernelSize.ToTuple(),
            strides: conv2d.Stride.ToTuple(),
            padding: conv2d.Padding.Dim1 == 0 && conv2d.Padding.Dim2 == 0 ? "valid" : "same");
        }

        /***************************************************/

        public static k.Layers.Dense ToBHoM(this Linear linear)
        {
            return new k.Layers.Dense(linear.FeaturesOut);
        }

        /***************************************************/

        public static k.Layers.GRU ToBHoM(this GRU gru)
        {
            return new k.Layers.GRU(
                units: gru.HiddenSize,
                recurrent_dropout: (float)gru.Dropout);
        }

        /***************************************************/

        public static k.Layers.LSTM ToBHoM(this LSTM lstm)
        {
            return new k.Layers.LSTM(
                units: lstm.HiddenSize,
                recurrent_dropout: (float)lstm.Dropout);
        }

        /***************************************************/

        public static k.Layers.MaxPooling2D ToBHoM(this MaxPooling2d maxPool2d)
        {
            return new k.Layers.MaxPooling2D(
                pool_size: maxPool2d.KernelSize.ToTuple(),
                strides: maxPool2d.Stride.ToTuple(),
                padding: maxPool2d.Padding.Dim1 == 0 && maxPool2d.Padding.Dim2 == 0 ? "valid" : "same",
                data_format: "channel_first");
        }

        /***************************************************/

        public static k.Layers.Conv2DTranspose ToBHoM(this TransposedConvolution2d transposedConv2d)
        {
            return new k.Layers.Conv2DTranspose(
                filters: transposedConv2d.FeaturesOut,
                kernel_size: transposedConv2d.KernelSize.ToTuple(),
                strides: transposedConv2d.Stride.ToTuple(),
                padding: transposedConv2d.Padding.Dim1 == 0 && transposedConv2d.Padding.Dim2 == 0 ? "valid" : "same",
                dilation_rate: transposedConv2d.Dilation.ToTuple());
        }

        /***************************************************/
    }
}
