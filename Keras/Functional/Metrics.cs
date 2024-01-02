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
    /// A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the metrics parameter when a model is compiled
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Metrics : Base
    {
        static dynamic caller = Keras.keras.metrics;

        public static NDarray MSE(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mse", parameters));
        }

        public static NDarray MAE(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mae", parameters));
        }

        public static NDarray MAPE(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mape", parameters));
        }

        public static NDarray MSLE(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "msle", parameters));
        }

        public static NDarray Cosine(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "cosine", parameters));
        }

        public static NDarray BinaryAccuracy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "binary_accuracy", parameters));
        }

        public static NDarray CategoricalAccuracy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "categorical_accuracy", parameters));
        }

        public static NDarray SparseCategoricalAccuracy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_accuracy", parameters));
        }

        public static NDarray TopKCategoricalAccuracy(NDarray y_true, NDarray y_pred, int k = 5)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "top_k_categorical_accuracy", parameters));
        }

        public static NDarray SparseTopKCategoricalAccuracy(NDarray y_true, NDarray y_pred, int k = 5)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "sparse_top_k_categorical_accuracy", parameters));
        }
    }
}

