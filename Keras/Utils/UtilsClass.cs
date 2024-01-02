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
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Utils
{
    /// <summary>
    /// Provides a scope that changes to _GLOBAL_CUSTOM_OBJECTS cannot escape.
    /// Code within a with statement will be able to access custom objects by name.Changes to global custom objects persist within the enclosing with statement. 
    /// At end of the with statement, global custom objects are reverted to state at beginning of the with statement.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class CustomObjectScope : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CustomObjectScope"/> class.
        /// </summary>
        public CustomObjectScope()
        {
            PyInstance = Keras.keras.utils.CustomObjectScope;
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="PyObject"/> to <see cref="CustomObjectScope"/>.
        /// </summary>
        /// <param name="py">The py.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator CustomObjectScope(PyObject py)
        {
            var obj = new CustomObjectScope();
            obj.PyInstance = py;

            return obj;
        }
    }

    /// <summary>
    /// Representation of HDF5 dataset to be used instead of a Numpy array.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class HDF5Matrix : Base
    {
        /// <summary>
        /// Prevents a default instance of the <see cref="HDF5Matrix"/> class from being created.
        /// </summary>
        private HDF5Matrix()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="HDF5Matrix" /> class.
        /// </summary>
        /// <param name="datapath">string, path to a HDF5 file</param>
        /// <param name="dataset">string, name of the HDF5 dataset in the file specified in datapath</param>
        /// <param name="start">int, start of desired slice of the specified dataset</param>
        /// <param name="end"> int, end of desired slice of the specified dataset.</param>
        /// <param name="normalizer">function to be called on data when retrieved</param>
        public HDF5Matrix(string datapath, string dataset, int start = 0, int? end = null, EventHandler normalizer = null)
        {
            PyInstance = Keras.keras.utils.HDF5Matrix;
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="PyObject"/> to <see cref="CustomObjectScope"/>.
        /// </summary>
        /// <param name="py">The py.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator HDF5Matrix(PyObject py)
        {
            var obj = new HDF5Matrix();
            obj.PyInstance = py;

            return obj;
        }
    }

    /// <summary>
    /// Base object for fitting to a sequence of data, such as a dataset.
    /// Sequence are a safer way to do multiprocessing. This structure guarantees that the network will only train once on each sample per epoch which is not the case with generators.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public partial class Sequence : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Sequence"/> class.
        /// </summary>
        public Sequence()
        {
            PyInstance = Keras.keras.utils.Sequence;
            //Init();
        }

        internal Sequence(PyObject py)
        {
            PyInstance = py;
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="PyObject"/> to <see cref="CustomObjectScope"/>.
        /// </summary>
        /// <param name="py">The py.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator Sequence(PyObject py)
        {
            var obj = new Sequence();
            obj.PyInstance = py;

            return obj;
        }

        public static implicit operator Sequence(NDarray x)
        {
            return new Sequence(x.PyObject);
        }

        public static implicit operator Sequence(KerasIterator py)
        {
            var obj = new Sequence();
            obj.PyInstance = py.PyObject;

            return obj;
        }
    }
}

