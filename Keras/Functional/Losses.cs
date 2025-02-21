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

using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    /// <summary>
    /// A loss function (or objective function, or optimization score function) is one of the two parameters required to compile a model
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Losses : Base
    {
        static dynamic caller = Keras.keras.losses;

        /// <summary>
        /// Calculates the mean squared error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanSquaredError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mean_squared_error", parameters));
        }

        /// <summary>
        /// Calculates the mean absolute error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanAbsoluteError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_error", parameters));
        }

        /// <summary>
        /// Calculates the mean absolute percentage error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanAbsolutePercentageError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_percentage_error", parameters));
        }

        /// <summary>
        /// Calculates the mean squared log error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanSquaredLogarithmicError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mean_squared_logarithmic_error", parameters));
        }

        /// <summary>
        /// Calculates the Square Hinge
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray SquaredHinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "squared_hinge", parameters));
        }

        /// <summary>
        /// Calculates the Hinge error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray Hinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "hinge", parameters));
        }

        /// <summary>
        /// Calculates the categorial hinge.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray CategoricalHinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "categorical_hinge", parameters));
        }

        /// <summary>
        /// Logarithm of the hyperbolic cosine of the prediction error.
        /// log(cosh(x)) is approximately equal to(x** 2) / 2 for small x and to abs(x) - log(2) for large x.This means that 'logcosh' works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray LogCosh(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "logcosh", parameters));
        }

        /// <summary>
        /// Categoricals the crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray CategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            parameters["with_logits"] = true;
            return new NDarray(InvokeStaticMethod(caller, "categorical_crossentropy", parameters));
        }

        /// <summary>
        /// Categoricals the crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray NegLogLikelihood(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            parameters["with_logits"] = false;
            return new NDarray(InvokeStaticMethod(caller, "categorical_crossentropy", parameters));
        }

        /// <summary>
        /// Sparses the categorical crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray SparseCategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_crossentropy", parameters));
        }

        /// <summary>
        /// Binaries the crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray BinaryCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "binary_crossentropy", parameters));
        }

        /// <summary>
        /// Crossentropy with sigmoid activation
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray BCEWithLogits(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            parameters["with_logits"] = true;
            return new NDarray(InvokeStaticMethod(caller, "binary_crossentropy", parameters));
        }

        /// <summary>
        /// Kullbacks the leibler divergence.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray KullbackLeiblerDivergence(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "kullback_leibler_divergence", parameters));
        }

        /// <summary>
        /// Poissons the specified y true.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray Poisson(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "poisson", parameters));
        }

        /// <summary>
        /// Cosines the proximity.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray CosineProximity(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "cosine_proximity", parameters));
        }
    }
}

