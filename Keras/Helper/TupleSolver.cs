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

namespace Keras.Helper
{
    public static class TupleSolver
    {
        public static T[] TupleToList<T>(PyObject obj)
        {
            List<T> result = new List<T>();
            GetTTupleList(new PyIter(obj), ref result);

            return result.ToArray();
        }

        private static void GetTTupleList<T>(PyObject obj, ref List<T> result)
        {
            PyIter iter = new PyIter(obj);

            while (iter.MoveNext())
            {
                var r = iter.Current.ToPython();
                if (PyTuple.IsTupleType(r))
                {
                    GetTTupleList<T>(r, ref result);
                    continue;
                }

                switch (typeof(T).Name)
                {
                    case "Single":
                    case "Double":
                    case "Int32":
                    case "Int64":
                    case "UInt32":
                    case "UInt64":
                    case "Byte":
                    case "Object":
                    case "String":
                    case "SByte":
                        result.Add(r.As<T>());
                        break;
                    default:
                        break;
                }
            }
        }

        public static NDarray[] TupleToList(PyObject obj)
        {
            PyIter iter = new PyIter(obj);
            List<NDarray> result = new List<NDarray>();
            GetNdListFromTuple(new PyIter(obj), ref result);

            return result.ToArray();
        }

        private static void GetNdListFromTuple(PyObject obj, ref List<NDarray> result)
        {
            PyIter iter = new PyIter(obj);
            
            while (iter.MoveNext())
            {
                var r = iter.Current.ToPython();

                if (PyTuple.IsTupleType(r))
                {
                    GetNdListFromTuple(r, ref result);
                    continue;
                }

                result.Add(new NDarray(r));
            }
        }
    }
}

