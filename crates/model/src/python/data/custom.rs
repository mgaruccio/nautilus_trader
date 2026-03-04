// -------------------------------------------------------------------------------------------------
//  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
//  https://nautechsystems.io
//
//  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
//  You may not use this code except in compliance with the License.
//  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// -------------------------------------------------------------------------------------------------

//! Python bindings for [`CustomData`].

use nautilus_core::python::to_pyvalue_err;
use pyo3::{prelude::*, types::PyAny};

use crate::data::{
    CustomData, DataType,
    custom::{PythonCustomDataWrapper, parse_custom_data_from_json_bytes},
    registry::try_extract_from_py,
};

#[pymethods]
impl CustomData {
    #[new]
    #[pyo3(signature = (data_type, data))]
    fn py_new(py: Python<'_>, data_type: DataType, data: Bound<'_, PyAny>) -> PyResult<Self> {
        let type_name = data_type.type_name();
        if let Some(arc) = try_extract_from_py(type_name, &data) {
            return Ok(Self::new(arc, data_type));
        }
        let wrapper = PythonCustomDataWrapper::new(py, &data)?;
        Ok(Self::new(std::sync::Arc::new(wrapper), data_type))
    }

    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.data.to_pyobject(py)
    }

    #[getter]
    fn data_type(&self) -> DataType {
        self.data_type.clone()
    }

    #[getter]
    fn ts_event(&self) -> u64 {
        self.data.ts_event().as_u64()
    }

    #[getter]
    fn ts_init(&self) -> u64 {
        self.data.ts_init().as_u64()
    }

    /// Serializes this CustomData to JSON bytes for roundtrip with from_json_bytes.
    fn to_json_bytes(&self) -> PyResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(to_pyvalue_err)
    }

    /// Deserializes CustomData from JSON bytes (full CustomData format).
    #[classmethod]
    #[pyo3(name = "from_json_bytes")]
    fn py_from_json_bytes_py(
        _cls: pyo3::Bound<'_, pyo3::types::PyType>,
        bytes: &[u8],
    ) -> PyResult<Self> {
        parse_custom_data_from_json_bytes(bytes).map_err(to_pyvalue_err)
    }
}
