//! IndexedDB helpers for cache persistence

use wasm_bindgen::prelude::*;

pub(crate) const IDB_NAME: &str = "hermes-cache";
pub(crate) const IDB_STORE: &str = "slices";
pub(crate) const IDB_VERSION: u32 = 1;

/// Generate a cache key from a URL
pub(crate) fn cache_key(url: &str) -> String {
    format!("cache:{}", url.trim_end_matches('/'))
}

/// Open the IndexedDB database
pub(crate) async fn open_idb() -> Result<web_sys::IdbDatabase, JsValue> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
    let idb_factory = window
        .indexed_db()?
        .ok_or_else(|| JsValue::from_str("IndexedDB not available"))?;

    let open_request = idb_factory.open_with_u32(IDB_NAME, IDB_VERSION)?;

    // Handle upgrade needed (create object store)
    let onupgradeneeded =
        wasm_bindgen::closure::Closure::once(move |event: web_sys::IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request: web_sys::IdbOpenDbRequest = target.dyn_into().unwrap();
            let db: web_sys::IdbDatabase = request.result().unwrap().dyn_into().unwrap();

            if !db.object_store_names().contains(IDB_STORE) {
                db.create_object_store(IDB_STORE).unwrap();
            }
        });
    open_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
    onupgradeneeded.forget();

    // Wait for success
    let (tx, rx) = futures_channel::oneshot::channel();
    let tx = std::cell::RefCell::new(Some(tx));

    let onsuccess = wasm_bindgen::closure::Closure::once(move |_event: web_sys::Event| {
        if let Some(tx) = tx.borrow_mut().take() {
            let _ = tx.send(());
        }
    });
    open_request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
    onsuccess.forget();

    let onerror = wasm_bindgen::closure::Closure::once(move |_event: web_sys::Event| {
        web_sys::console::error_1(&"IndexedDB open error".into());
    });
    open_request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
    onerror.forget();

    rx.await
        .map_err(|_| JsValue::from_str("IndexedDB open cancelled"))?;

    open_request.result()?.dyn_into()
}

/// Store data in IndexedDB
pub(crate) async fn idb_put(key: &str, data: &[u8]) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;

    let db = open_idb().await?;
    let tx = db.transaction_with_str_and_mode(IDB_STORE, web_sys::IdbTransactionMode::Readwrite)?;
    let store = tx.object_store(IDB_STORE)?;

    let js_data = js_sys::Uint8Array::from(data);
    store.put_with_key(&js_data, &JsValue::from_str(key))?;

    // Wait for transaction to complete
    let (tx_done, rx_done) = futures_channel::oneshot::channel();
    let tx_done = std::cell::RefCell::new(Some(tx_done));

    let oncomplete = wasm_bindgen::closure::Closure::once(move |_event: web_sys::Event| {
        if let Some(tx) = tx_done.borrow_mut().take() {
            let _ = tx.send(());
        }
    });
    tx.set_oncomplete(Some(oncomplete.as_ref().unchecked_ref()));
    oncomplete.forget();

    rx_done
        .await
        .map_err(|_| JsValue::from_str("IndexedDB put cancelled"))?;
    Ok(())
}

/// Get data from IndexedDB
pub(crate) async fn idb_get(key: &str) -> Result<Option<Vec<u8>>, JsValue> {
    use wasm_bindgen::JsCast;

    let db = open_idb().await?;
    let tx = db.transaction_with_str(IDB_STORE)?;
    let store = tx.object_store(IDB_STORE)?;

    let request = store.get(&JsValue::from_str(key))?;

    let (tx_done, rx_done) = futures_channel::oneshot::channel();
    let tx_done = std::cell::RefCell::new(Some(tx_done));

    let onsuccess = wasm_bindgen::closure::Closure::once(move |event: web_sys::Event| {
        let target = event.target().unwrap();
        let request: web_sys::IdbRequest = target.dyn_into().unwrap();
        let result = request.result().unwrap();

        let data = if result.is_undefined() || result.is_null() {
            None
        } else {
            let array: js_sys::Uint8Array = result.dyn_into().unwrap();
            Some(array.to_vec())
        };

        if let Some(tx) = tx_done.borrow_mut().take() {
            let _ = tx.send(data);
        }
    });
    request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
    onsuccess.forget();

    rx_done
        .await
        .map_err(|_| JsValue::from_str("IndexedDB get cancelled"))
}

/// Delete data from IndexedDB
pub(crate) async fn idb_delete(key: &str) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;

    let db = open_idb().await?;
    let tx = db.transaction_with_str_and_mode(IDB_STORE, web_sys::IdbTransactionMode::Readwrite)?;
    let store = tx.object_store(IDB_STORE)?;

    store.delete(&JsValue::from_str(key))?;

    let (tx_done, rx_done) = futures_channel::oneshot::channel();
    let tx_done = std::cell::RefCell::new(Some(tx_done));

    let oncomplete = wasm_bindgen::closure::Closure::once(move |_event: web_sys::Event| {
        if let Some(tx) = tx_done.borrow_mut().take() {
            let _ = tx.send(());
        }
    });
    tx.set_oncomplete(Some(oncomplete.as_ref().unchecked_ref()));
    oncomplete.forget();

    rx_done
        .await
        .map_err(|_| JsValue::from_str("IndexedDB delete cancelled"))?;
    Ok(())
}
