# Expensit Backend — API Documentation

> **Stack:** FastAPI · PostgreSQL · Firebase Auth · OpenAI GPT-4o · Google Cloud Vision · Sentence Transformers
> **Base URL:** `http://<host>:<port>`
> **OpenAPI docs:** `GET /docs` (Swagger UI), `GET /redoc` (ReDoc)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Products](#products)
   - [GET /products](#get-products)
   - [GET /product/{product_id}](#get-productproduct_id)
   - [POST /product/add-product](#post-productadd-product)
   - [POST /product/add-products-bulk](#post-productadd-products-bulk)
   - [POST /product/add-price](#post-productadd-price)
   - [POST /products/compare](#post-productscompare)
   - [POST /products/compare/bulk](#post-productscomparebulk)
3. [Receipts](#receipts)
   - [POST /receipt/add](#post-receiptadd)
   - [GET /receipts](#get-receipts)
   - [GET /receipt/{receipt_id}](#get-receiptreceipt_id)
4. [AI / OpenAI](#ai--openai)
   - [POST /openai/analyze-image](#post-openaianalyze-image)
   - [POST /openai/chat](#post-openaichat)
   - [DELETE /openai/chat/clear](#delete-openaichatclear)
5. [Data Models](#data-models)
6. [Error Responses](#error-responses)
7. [Environment & Configuration](#environment--configuration)

---

## Authentication

Protected endpoints (all `/openai/*` routes) require a **Firebase ID Token** passed as a Bearer token.

```
Authorization: Bearer <firebase_id_token>
```

The server validates the token using the Firebase Admin SDK. The decoded token contains the `uid` field (Firebase user ID), which is used to scope data access per user.

**Obtaining a token:**
Use the Firebase client SDK (`signInWithEmailAndPassword`, Google Sign-In, etc.) to get an ID token, then pass it in the `Authorization` header.

**Errors:**

| Code | Reason |
|------|--------|
| `401` | Missing, invalid, or expired Firebase token |
| `403` | Token valid but resource does not belong to this user |

---

## Products

### GET /products

Returns all products with their current prices.

**Authentication:** Not required

**Response `200 OK`**
```json
[
  {
    "id": 1,
    "name": "Mleko 3.2% 1L",
    "price": 3.49,
    "description": null
  },
  {
    "id": 2,
    "name": "Chleb tostowy",
    "price": 5.99,
    "description": "pieczywo"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `integer` | Product ID |
| `name` | `string` | Product name |
| `price` | `float` | Current price (PLN) |
| `description` | `string \| null` | Optional product description |

---

### GET /product/{product_id}

Returns a single product including its embedding vector.

**Authentication:** Not required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `integer` | Product ID |

**Response `200 OK`**
```json
{
  "id": 1,
  "name": "Mleko 3.2% 1L",
  "price": 3.49,
  "description": null,
  "embedding": [0.0234, -0.1123, 0.0891, "..."]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `integer` | Product ID |
| `name` | `string` | Product name |
| `price` | `float \| null` | Current price |
| `description` | `string \| null` | Optional description |
| `embedding` | `float[]` | 384-dimensional semantic embedding vector |

**Response `404 Not Found`** — product does not exist.

---

### POST /product/add-product

Adds a single product to the database. Automatically generates a semantic embedding if one is not provided.

**Authentication:** Not required

**Request Body**
```json
{
  "name": "Masło extra 200g",
  "price": 7.99,
  "description": "nabiał",
  "embedding": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Product name |
| `price` | `float` | Yes | Product price (PLN) |
| `description` | `string` | No | Optional description |
| `embedding` | `float[]` | No | Pre-computed 384-dim embedding; generated automatically if omitted |

**Response `200 OK`**
```json
{
  "id": 42,
  "name": "Masło extra 200g",
  "status": "inserted"
}
```

---

### POST /product/add-products-bulk

Adds multiple products in a single request.

**Authentication:** Not required

**Request Body** — array of `ProductIn` objects (see [POST /product/add-product](#post-productadd-product))

```json
[
  { "name": "Jogurt naturalny", "price": 2.49 },
  { "name": "Ser żółty 200g", "price": 8.99, "description": "nabiał" }
]
```

**Response `200 OK`**
```json
{
  "total_processed": 2,
  "successful_inserts": 2,
  "errors": 0,
  "results": [
    { "id": 43, "name": "Jogurt naturalny", "status": "inserted", "error": null },
    { "id": 44, "name": "Ser żółty 200g", "status": "inserted", "error": null }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_processed` | `integer` | Total number of items in the request |
| `successful_inserts` | `integer` | Number of successfully inserted products |
| `errors` | `integer` | Number of failed inserts |
| `results[].status` | `"inserted" \| "error"` | Per-item status |
| `results[].error` | `string \| null` | Error message if insertion failed |

---

### POST /product/add-price

Adds or updates the price of an existing product.

**Authentication:** Not required

**Request Body**
```json
{
  "id": 42,
  "price": 8.49
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `integer` | Yes | Product ID |
| `price` | `float` | Yes | New price (PLN) |

**Response `200 OK`**
```json
{
  "id": 42,
  "price": 8.49,
  "status": "updated or inserted"
}
```

---

### POST /products/compare

Checks whether a product already exists in the database and whether its price has changed. Matching uses a **hybrid similarity algorithm**: semantic cosine similarity (sentence-transformers `all-MiniLM-L6-v2`) combined with fuzzy string matching. The similarity threshold is **0.80**.

**Authentication:** Not required

**Request Body**
```json
{
  "name": "Mleko UHT 3.2%",
  "price": 3.69,
  "description": null,
  "embedding": null
}
```

**Response `200 OK`**
```json
{
  "exists": true,
  "id": 1,
  "name": "Mleko 3.2% 1L",
  "old_price": 3.49,
  "new_price": 3.69,
  "price_changed": true,
  "similarity": 0.93
}
```

| Field | Type | Description |
|-------|------|-------------|
| `exists` | `boolean` | Whether a similar product was found |
| `id` | `integer \| null` | ID of matched product |
| `name` | `string \| null` | Name of matched product |
| `old_price` | `float \| null` | Current price in the database |
| `new_price` | `float` | Price from the request |
| `price_changed` | `boolean \| null` | `true` if prices differ |
| `similarity` | `float` | Similarity score (0–1) |

---

### POST /products/compare/bulk

Batch version of `/products/compare`. Uses vectorized numpy operations for efficient multi-product comparison in a single database round-trip.

**Authentication:** Not required

**Request Body** — array of `ProductIn` objects
```json
[
  { "name": "Mleko UHT 3.2%", "price": 3.69 },
  { "name": "Chleb tostowy", "price": 5.99 }
]
```

**Response `200 OK`** — array of comparison results (same schema as `/products/compare`)
```json
[
  {
    "exists": true,
    "id": 1,
    "name": "Mleko 3.2% 1L",
    "old_price": 3.49,
    "new_price": 3.69,
    "price_changed": true,
    "similarity": 0.93
  },
  {
    "exists": true,
    "id": 2,
    "name": "Chleb tostowy",
    "old_price": 5.99,
    "new_price": 5.99,
    "price_changed": false,
    "similarity": 1.0
  }
]
```

**Performance notes:**
- All products are compared in a single database query
- Embeddings are computed in a single batched inference call
- Cosine similarity is computed via vectorized numpy operations (no per-product loop)
- Per-item: only the top 10 candidates by string similarity undergo full embedding comparison

---

## Receipts

### POST /receipt/add

Adds a new receipt along with its line items. Before linking products:

1. Each product name is matched against the existing product catalog using semantic similarity.
2. If a match above threshold (0.80) is found, the existing product is reused (and its price updated if changed).
3. If no match is found, a new product is created.

**Authentication:** Not required

**Request Body**
```json
{
  "store_name": "Biedronka",
  "document_type": "PARAGON",
  "payment_method": "KARTA",
  "total_amount": 45.67,
  "purchase_date": "2024-03-15T14:32:00",
  "products": [
    {
      "product_id": null,
      "product_name": "Mleko UHT 3.2%",
      "quantity": 2,
      "unit_price": 3.49,
      "total_price": 6.98,
      "purchase_date": null
    },
    {
      "product_id": null,
      "product_name": "Chleb tostowy",
      "quantity": 1,
      "unit_price": 5.99,
      "total_price": 5.99,
      "purchase_date": null
    }
  ]
}
```

**Receipt fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_name` | `string` | Yes | Name of the store |
| `document_type` | `string` | Yes | e.g. `PARAGON`, `FAKTURA`, `RACHUNEK` |
| `payment_method` | `string` | No | e.g. `KARTA`, `GOTÓWKA`, `BLIK` |
| `total_amount` | `float` | Yes | Total receipt amount (PLN) |
| `purchase_date` | `datetime` | Yes | ISO 8601 datetime of purchase |
| `products` | `array` | Yes | List of line items (see below) |

**Product line item fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `product_id` | `integer` | No | If known, skip product matching |
| `product_name` | `string` | No | Used for semantic matching if `product_id` is null |
| `quantity` | `float` | Yes | Quantity purchased |
| `unit_price` | `float` | Yes | Price per unit (PLN) |
| `total_price` | `float` | Yes | Total price for line item (PLN) |
| `purchase_date` | `datetime` | No | Override purchase date for this item |

**Response `200 OK`**
```json
{
  "id": 101,
  "store_name": "Biedronka",
  "document_type": "PARAGON",
  "payment_method": "KARTA",
  "total_amount": 45.67,
  "purchase_date": "2024-03-15T14:32:00",
  "products": [
    {
      "product_id": 1,
      "product_name": "Mleko 3.2% 1L",
      "quantity": 2,
      "unit_price": 3.49,
      "total_price": 6.98,
      "purchase_date": null
    }
  ]
}
```

---

### GET /receipts

Returns a paginated list of all receipts with their line items.

**Authentication:** Not required

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `integer` | `50` | Max number of receipts to return |
| `offset` | `integer` | `0` | Number of receipts to skip |

**Example:** `GET /receipts?limit=10&offset=20`

**Response `200 OK`** — array of `ReceiptResponse` objects (same schema as [POST /receipt/add](#post-receiptadd) response)

---

### GET /receipt/{receipt_id}

Returns a single receipt with all its line items.

**Authentication:** Not required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `receipt_id` | `integer` | Receipt ID |

**Response `200 OK`** — `ReceiptResponse` object

**Response `404 Not Found`**
```json
{ "detail": "Receipt not found" }
```

---

## AI / OpenAI

All endpoints in this section require Firebase authentication.

---

### POST /openai/analyze-image

Analyzes a receipt image using the Google Cloud Vision OCR pipeline followed by GPT-4o structuring. The image must exist in Firebase Storage under the authenticated user's path (`users/{uid}/...`).

**Authentication:** Required (Firebase Bearer token)

**Request Body**
```json
{
  "image_path": "users/abc123uid/receipts/biedronka_2024-03-15.jpg"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_path` | `string` | Yes | Path in Firebase Storage. Must start with `users/{uid}/` |

**Processing pipeline:**
1. Token verified → `uid` extracted
2. Image downloaded from Firebase Storage
3. Google Cloud Vision OCR extracts raw text
4. Raw OCR text + detailed prompt sent to GPT-4o
5. Structured JSON returned

**Response `200 OK`**
```json
{
  "status": "success",
  "analysis": {
    "Document_Type": "PARAGON",
    "Title": "Zakupy spożywcze Biedronka",
    "Category": "FOOD",
    "Store_Name": "Biedronka",
    "Store_Address": "ul. Kwiatowa 5, 00-001 Warszawa",
    "Tax_Identification_Number": "123-456-78-90",
    "Receipt_Number": "2024/03/15/001234",
    "Items": [
      {
        "Original_Item_Title": "MLEKO UHT 3,2% 1L",
        "Item_Title": "Mleko UHT 3.2% 1L",
        "Item_Price": "3.49 PLN",
        "Item_Category": "FOOD",
        "Item_Quantity": 2
      }
    ],
    "Total_Cost_Without_Tax": "37.13 PLN",
    "Tax_Classified_Sale": "A",
    "Tax_Rate": "8%",
    "Tax_Amount": "2.98 PLN",
    "Payment_Method": "KARTA",
    "Payment_Amount_in_Cash": "0.00 PLN",
    "Receipt_Footer": "Dziękujemy za zakupy!",
    "Date_and_Time": "2024-03-15 14:32",
    "System_Number": "SYS-001",
    "Double_Price": 45.67
  },
  "processing_time": "2.34s",
  "method": "vision_ocr"
}
```

**Analysis object fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Document_Type` | `string` | `PARAGON` / `FAKTURA` / `RACHUNEK` |
| `Title` | `string` | Short receipt description (max 50 chars) |
| `Category` | `string` | Main expense category (see categories below) |
| `Store_Name` | `string` | Store/merchant name |
| `Store_Address` | `string` | Store address |
| `Tax_Identification_Number` | `string` | NIP (Polish tax ID) |
| `Receipt_Number` | `string` | Receipt/invoice number |
| `Items` | `array` | Line items (see below) |
| `Total_Cost_Without_Tax` | `string` | Net amount string |
| `Tax_Classified_Sale` | `string` | VAT classification (`A`, `B`, etc.) |
| `Tax_Rate` | `string` | VAT rate (e.g. `23%`, `8%`, `5%`) |
| `Tax_Amount` | `string` | VAT amount string |
| `Payment_Method` | `string` | `GOTÓWKA` / `KARTA` / `BLIK` / `nie podano` |
| `Payment_Amount_in_Cash` | `string` | Cash tender amount |
| `Receipt_Footer` | `string` | Footer text from the receipt |
| `Date_and_Time` | `string` | `YYYY-MM-DD HH:MM` format |
| `System_Number` | `string` | POS system reference number |
| `Double_Price` | `float` | Total amount as a number (PLN) |

**Item fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Original_Item_Title` | `string` | Raw item name from OCR |
| `Item_Title` | `string` | Normalized item name |
| `Item_Price` | `string` | Price string (e.g. `"3.49 PLN"`) |
| `Item_Category` | `string` | Item-level category |
| `Item_Quantity` | `float` | Quantity |

**Supported categories:**
`FOOD` · `TRANSPORT` · `HEALTH` · `ENTERTAINMENT` · `ESSENTIALS` · `BEAUTY` · `SPORTS` · `EDUCATION` · `PETS` · `GARDEN`

**Error responses:**

| Code | Reason |
|------|--------|
| `401` | Missing or invalid Firebase token |
| `403` | Image path does not belong to authenticated user |
| `404` | Image not found in Firebase Storage |
| `400` | OCR returned no text (blank or unreadable image) |

---

### POST /openai/chat

Streaming AI chat endpoint for expense analysis and financial Q&A. Responses are returned as **Server-Sent Events (SSE)**.

**Authentication:** Required (Firebase Bearer token)

**Request Body**
```json
{
  "message": "Ile wydałem na jedzenie w tym miesiącu?",
  "products": {
    "receipts": [...],
    "summary": {...}
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | `string` | Yes | User's message |
| `products` | `dict` | Yes | Context data (receipts, summaries) injected into the system prompt |

**Response** — `text/event-stream` (SSE)

```
data: {"type": "content", "text": "W tym "}
data: {"type": "content", "text": "miesiącu "}
data: {"type": "content", "text": "wydałeś 234.50 PLN na jedzenie."}
data: {"type": "done"}
```

**SSE event types:**

| `type` | Description |
|--------|-------------|
| `content` | Partial text chunk from the model |
| `done` | Stream complete |
| `error` | Error occurred; `message` field contains description |

**Behavior:**
- Conversation history from the last **20 messages** is loaded from Firestore
- `products` context is embedded in the system prompt at each request
- Each message pair (user + assistant) is saved asynchronously to Firestore
- Uses OpenAI Assistants API with persistent threads (one thread per user)

**Firestore collections used:**
- `users/{uid}/messages` — conversation history
- `threads` — maps `uid` to OpenAI thread ID

---

### DELETE /openai/chat/clear

Clears the conversation history for the authenticated user and creates a new OpenAI thread.

**Authentication:** Required (Firebase Bearer token)

**Request Body:** none

**Response `200 OK`**
```json
{
  "status": "success",
  "message": "Chat history cleared",
  "old_thread_id": "thread_abc123",
  "new_thread_id": "thread_xyz789"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `old_thread_id` | `string` | Previous OpenAI thread ID |
| `new_thread_id` | `string` | Newly created OpenAI thread ID |

---

## Data Models

### ProductIn

```typescript
interface ProductIn {
  name: string;
  price: number;
  description?: string | null;
  embedding?: number[] | null;  // 384-dimensional vector
}
```

### ReceiptData

```typescript
interface ReceiptData {
  store_name: string;
  document_type: string;
  payment_method?: string | null;
  total_amount: number;
  purchase_date: string;  // ISO 8601
  products: ReceiptProductCreate[];
}

interface ReceiptProductCreate {
  product_id?: number | null;
  product_name?: string | null;
  quantity: number;
  unit_price: number;
  total_price: number;
  purchase_date?: string | null;
}
```

### ProductResponse

```typescript
interface ProductResponse {
  id: number;
  name: string;
  price: number | null;
  description: string | null;
  embedding: number[];
}
```

### ReceiptResponse

```typescript
interface ReceiptResponse {
  id: number;
  store_name: string;
  document_type: string;
  payment_method: string | null;
  total_amount: number;
  purchase_date: string;  // ISO 8601
  products: ReceiptProductCreate[];
}
```

---

## Error Responses

All errors follow the standard FastAPI format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

| HTTP Code | When |
|-----------|------|
| `400 Bad Request` | Invalid input or OCR found no text |
| `401 Unauthorized` | Missing or invalid Firebase token |
| `403 Forbidden` | Accessing another user's resource |
| `404 Not Found` | Resource (product, receipt, image) does not exist |
| `422 Unprocessable Entity` | Request body failed Pydantic validation |
| `500 Internal Server Error` | Unexpected server-side error |

---

## Environment & Configuration

### Required environment variables (`.env`)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (`sk-proj-...`) |
| `DATABASE_URL` | PostgreSQL connection string (e.g. `postgresql://user:pass@host:port/db`) |
| `GPT_PROMPT` | System prompt for receipt parsing (can also be loaded from `prompt.txt`) |
| `QDRANT_API_KEY` | Qdrant vector DB API key |

### Firebase

Firebase service account credentials are loaded from:
`config/expensit-10546-firebase-adminsdk-q25nr-1769d00b3c.json`

Firebase Storage bucket: `gs://expensit-10546.appspot.com/images`

### Database schema

**Table: `products`**

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` | Primary key, auto-increment |
| `name` | `VARCHAR` | Product name |
| `embedding` | `JSON` | 384-dim float array |
| `description` | `VARCHAR` | Optional |

**Table: `prices`**

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` | FK → `products.id` |
| `price` | `FLOAT` | Current price (PLN) |

**Table: `receipts`**

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` | Primary key |
| `store_name` | `VARCHAR` | |
| `document_type` | `VARCHAR` | |
| `payment_method` | `VARCHAR` | Nullable |
| `total_amount` | `FLOAT` | |
| `purchase_date` | `DATETIME` | |
| `created_at` | `DATETIME` | Default: `now()` |

**Table: `receipt_products`**

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` | Primary key |
| `receipt_id` | `INTEGER` | FK → `receipts.id` |
| `product_id` | `INTEGER` | FK → `products.id` |
| `product_name` | `VARCHAR` | Snapshot of name at time of purchase |
| `quantity` | `INTEGER` | |
| `unit_price` | `FLOAT` | |
| `total_price` | `FLOAT` | |
| `created_at` | `DATETIME` | |

---

## Semantic Similarity — How It Works

Product matching uses a **hybrid similarity score**:

```
score = α × cosine_similarity(embedding1, embedding2)
      + (1-α) × fuzzy_string_similarity(name1, name2)
      + 0.05 (bonus if first token matches)
```

Default `α = 0.5`. The match threshold is **0.80**.

- Embeddings are generated by `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Fuzzy string matching uses `difflib.SequenceMatcher`
- The model is cached globally; CUDA is used if available

This approach prevents duplicate products when the same item appears on receipts with slightly different names (e.g. `"MLEKO UHT 3,2% 1L"` vs `"Mleko 3.2% 1L"`).
