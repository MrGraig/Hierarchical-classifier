## Project
This project implements 3 classifier structures: 
- Local Classifier par Parent Node (LCPN) - each parent node receives one multiclass classifier.
- Local Classifier per Node (LCN) - training one multiclass classifier for each level.
- Flat Classifier ("flat" classifier) - training on concatenated data.

The implementation of each of them is in the app/model/analyze.ipynb folder. 

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` for managing Python packages
- Docker (optional, if using Docker)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fastapi-text-classification.git
    cd fastapi-text-classification
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure your model files (`model.pkl`, `encoder.pkl`, `vectorizer.pkl`) are located in the `model/` directory.

### Running the Service

Run the FastAPI application locally:

```bash
uvicorn main:app --reload
```

### Docker
1. Building the Docker Image:
    ```bash
    docker build -t fastapi-app .
    ```
    
2. Running the Docker Container
   ```bash
   docker run -d -p 8000:8000 fastapi-app
   ```

