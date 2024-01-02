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

using Keras.Helper;
using Keras.Models;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Text;

namespace Keras.Applications
{
    /// <summary>
    /// Model base class for keras applications
    /// </summary>
    /// <seealso cref="Keras.Models.Model" />
    public class AppModelBase : Model
    {
        internal dynamic caller;

        /// <summary>
        /// Initializes a new instance of the <see cref="AppModelBase"/> class.
        /// </summary>
        /// <param name="_caller">The caller.</param>
        public AppModelBase(dynamic _caller)
        {
            caller = _caller;
        }

        /// <summary>
        /// Decodes the predictions.
        /// </summary>
        /// <param name="preds">The preds.</param>
        /// <param name="top">The top.</param>
        /// <returns></returns>
        public ImageNetPrediction[] DecodePredictions(NDarray preds, int top = 3)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["preds"] = preds;
            parameters["top"] = top;
            var predobj = (PyObject)InvokeStaticMethod(caller, "decode_predictions", parameters);
            var d = predobj.ToString();
            var list = TupleSolver.TupleToList<object>(predobj[0]);
            List<ImageNetPrediction> predictions = new List<ImageNetPrediction>();
            for (int i = 0; i < list.Length; i = i++)
            {
                ImageNetPrediction pred = new ImageNetPrediction()
                {
                    WordID = list[i].ToString(),
                    Word = list[i + 1].ToString(),
                    PredictedValue = Convert.ToSingle(list[i + 2].ToString()),
                };

                i = i + 3;

                predictions.Add(pred);
            }

            return predictions.ToArray();
        }

        /// <summary>
        /// Preprocesses the input.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public NDarray PreprocessInput(NDarray x, string data_format = "channels_last")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["data_format"] = data_format;
            //Parameters["mode"] = mode;
            return new NDarray((PyObject)InvokeStaticMethod(caller, "preprocess_input", parameters));
        }
    }
}

