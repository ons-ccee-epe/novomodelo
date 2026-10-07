//! Conversions between Python objects and `serde_json` values, the inverse of
//! [`crate::results::json_value_to_py`].
//!
//! The override map must be converted **under the GIL** (via [`pydict_to_json_map`])
//! before the solver lifecycle releases it with `py.detach`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::Map;
use serde_json::Value;

/// Convert an arbitrary JSON-compatible Python object into a [`serde_json::Value`].
///
/// # Errors
///
/// Returns [`PyValueError`] when `obj` is an unsupported type, a `float` is
/// `NaN`/infinite, an integer is outside `i64`/`u64`, or a `dict` key is not a `str`.
pub fn py_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }

    // `bool` must be checked before `int`: Python `bool` is an `int` subclass.
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(serde_json::Value::Bool(b.is_true()));
    }

    if let Ok(i) = obj.cast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            return Ok(serde_json::Value::Number(v.into()));
        }
        if let Ok(v) = i.extract::<u64>() {
            return Ok(serde_json::Value::Number(v.into()));
        }
        return Err(PyValueError::new_err(
            "integer override value is out of the supported range (i64/u64)",
        ));
    }

    if let Ok(f) = obj.cast::<PyFloat>() {
        let v = f.value();
        let number = serde_json::Number::from_f64(v).ok_or_else(|| {
            PyValueError::new_err(
                "float override value (NaN or infinity) is not representable in JSON",
            )
        })?;
        return Ok(serde_json::Value::Number(number));
    }

    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(serde_json::Value::String(s.to_str()?.to_owned()));
    }

    if let Ok(list) = obj.cast::<PyList>() {
        let arr = list
            .iter()
            .map(|item| py_to_json_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let arr = tuple
            .iter()
            .map(|item| py_to_json_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(dict) = obj.cast::<PyDict>() {
        return Ok(serde_json::Value::Object(pydict_to_json_map(dict)?));
    }

    Err(PyValueError::new_err(format!(
        "unsupported override value type `{}`: expected one of None, bool, int, \
         float, str, list, tuple, or dict",
        obj.get_type().name()?,
    )))
}

/// Convert a Python `dict` into a [`serde_json::Map`].
///
/// # Errors
///
/// Returns [`PyValueError`] when a key is not a `str`, or when any value fails
/// [`py_to_json_value`] conversion.
pub fn pydict_to_json_map(dict: &Bound<'_, PyDict>) -> PyResult<Map<String, Value>> {
    let mut map = serde_json::Map::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_str = key.cast::<PyString>().map_err(|_| {
            PyValueError::new_err(format!(
                "override map keys must be strings, got `{}`",
                key.get_type()
                    .name()
                    .map_or_else(|_| "<unknown>".to_owned(), |n| n.to_string()),
            ))
        })?;
        map.insert(key_str.to_str()?.to_owned(), py_to_json_value(&value)?);
    }
    Ok(map)
}

// GIL-bound: a Rust `#[cfg(test)]` calling `Python::attach` cannot link the
// extension-module build (no embedded interpreter), so coverage lives in
// `tests/test_convert.py`.
