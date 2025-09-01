# ner_model_loader.py - NER model loader
import os
import json
import torch
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NERModel:
    """NER model wrapper class providing unified interface"""
    
    def __init__(self, model, tokenizer=None, model_type="transformers"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        
    def predict(self, text: str) -> List[Dict]:
        """Predict named entities in text"""
        return extract_entities_with_model(text, self.model, self.tokenizer, self.model_type)

# Global NER model instances
_ner_model = None
_tokenizer = None

def load_ner_model(model_dir: str = "transformer_ner_model"):
    """Load NER model - supports multiple model types"""
    global _ner_model, _tokenizer
    
    if not os.path.exists(model_dir):
        logger.error(f"NER model directory does not exist: {model_dir}")
        return None
    
    try:
        # Option 1: Transformers model
        if os.path.exists(os.path.join(model_dir, "config.json")):
            model, tokenizer = load_transformers_ner_model(model_dir)
            if model:
                _ner_model = model
                _tokenizer = tokenizer
                return NERModel(model, tokenizer, "transformers")
        
        # Option 2: spaCy model
        elif os.path.exists(os.path.join(model_dir, "meta.json")):
            model = load_spacy_ner_model(model_dir)
            if model:
                _ner_model = model
                return NERModel(model, None, "spacy")
        
        # Option 3: Custom PyTorch model
        elif os.path.exists(os.path.join(model_dir, "model.pth")):
            model = load_pytorch_ner_model(model_dir)
            if model:
                _ner_model = model
                return NERModel(model, None, "pytorch")
        
        else:
            logger.error(f"Cannot recognize NER model type: {model_dir}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading NER model: {e}")
        return None

def load_transformers_ner_model(model_dir: str):
    """Load Transformers NER model"""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
        
        logger.info(f"Loading Transformers NER model: {model_dir}")
        
        # Fix config.json if needed
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            if 'model_type' not in config_data:
                architectures = config_data.get('architectures', [])
                if architectures:
                    arch = architectures[0].lower()
                    if 'bert' in arch:
                        config_data['model_type'] = 'bert'
                    elif 'roberta' in arch:
                        config_data['model_type'] = 'roberta'
                    elif 'distilbert' in arch:
                        config_data['model_type'] = 'distilbert'
                    else:
                        config_data['model_type'] = 'bert'
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Fixed config.json, added model_type: {config_data['model_type']}")
        
        # Load tokenizer
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            logger.warning(f"Cannot load tokenizer, trying bert-base-uncased: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            except Exception as e2:
                logger.error(f"Cannot load fallback tokenizer: {e2}")
                return None, None
        
        # Load model
        model = None
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
        except Exception as e:
            logger.error(f"Direct loading failed, trying manual config: {e}")
            
            try:
                config = AutoConfig.from_pretrained(model_dir)
                model = AutoModelForTokenClassification.from_pretrained(
                    model_dir, 
                    config=config,
                    ignore_mismatched_sizes=True
                )
            except Exception as e2:
                logger.error(f"Manual loading also failed: {e2}")
                return None, None
        
        model.eval()
        
        logger.info("Transformers NER model loaded successfully")
        return model, tokenizer
        
    except ImportError:
        logger.error("Need to install transformers library: pip install transformers")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load Transformers model: {e}")
        return None, None

def load_spacy_ner_model(model_dir: str):
    """Load spaCy NER model"""
    try:
        import spacy
        
        logger.info(f"Loading spaCy NER model: {model_dir}")
        model = spacy.load(model_dir)
        
        logger.info("spaCy NER model loaded successfully")
        return model
        
    except ImportError:
        logger.error("Need to install spacy library: pip install spacy")
        return None
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        return None

def load_pytorch_ner_model(model_dir: str):
    """Load custom PyTorch NER model"""
    try:
        logger.info(f"Loading PyTorch NER model: {model_dir}")
        
        model_path = os.path.join(model_dir, "model.pth")
        model_state = torch.load(model_path, map_location='cpu')
        
        # This needs to be adjusted based on your specific model architecture
        # Example:
        # from your_model_class import YourNERModel
        # model = YourNERModel(config)
        # model.load_state_dict(model_state)
        
        logger.warning("PyTorch NER model loading requires custom implementation")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        return None

def extract_entities_with_model(text: str, model=None, tokenizer=None, model_type="transformers") -> List[Dict]:
    """Extract entities using NER model"""
    if model is None:
        global _ner_model, _tokenizer
        model = _ner_model
        tokenizer = _tokenizer
        model_type = "transformers"
    
    if model is None:
        logger.warning("NER model not loaded")
        return []
    
    try:
        if model_type == "transformers":
            return extract_entities_transformers(text, model, tokenizer)
        elif model_type == "spacy":
            return extract_entities_spacy(text, model)
        elif model_type == "pytorch":
            return extract_entities_pytorch(text, model)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return []
            
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []

def extract_entities_transformers(text: str, model, tokenizer) -> List[Dict]:
    """Extract entities using Transformers model"""
    try:
        from transformers import pipeline
        
        ner_pipeline = pipeline("ner", 
                               model=model, 
                               tokenizer=tokenizer,
                               aggregation_strategy="simple",
                               device=-1)
        
        entities = ner_pipeline(text)
        
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "text": entity["word"],
                "label": entity["entity_group"],
                "start": entity["start"],
                "end": entity["end"],
                "confidence": float(entity["score"])
            })
        
        return formatted_entities
        
    except Exception as e:
        logger.error(f"Transformers entity extraction failed: {e}")
        try:
            return extract_entities_transformers_manual(text, model, tokenizer)
        except Exception as e2:
            logger.error(f"Manual entity extraction also failed: {e2}")
            return []

def extract_entities_transformers_manual(text: str, model, tokenizer) -> List[Dict]:
    """Manual Transformers entity extraction (fallback when pipeline fails)"""
    try:
        import torch
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)
        
        id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}
        
        entities = []
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        current_entity = None
        for i, (token, pred_id, offset) in enumerate(zip(tokens, predicted_token_class_ids[0], offsets[0])):
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
                
            label = id2label.get(pred_id.item(), f"LABEL_{pred_id.item()}")
            confidence = float(predictions[0][i][pred_id].item())
            
            if label == "O" or label.startswith("O-"):
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            if label.startswith("B-") or (current_entity is None):
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    "text": token.replace("##", ""),
                    "label": label.replace("B-", "").replace("I-", ""),
                    "start": int(offset[0]),
                    "end": int(offset[1]),
                    "confidence": confidence
                }
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += token.replace("##", "")
                current_entity["end"] = int(offset[1])
                current_entity["confidence"] = (current_entity["confidence"] + confidence) / 2
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
        
    except Exception as e:
        logger.error(f"Manual entity extraction failed: {e}")
        return []

def extract_entities_spacy(text: str, model) -> List[Dict]:
    """Extract entities using spaCy model"""
    try:
        doc = model(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0
            })
        
        return entities
        
    except Exception as e:
        logger.error(f"spaCy entity extraction failed: {e}")
        return []

def extract_entities_pytorch(text: str, model) -> List[Dict]:
    """Extract entities using custom PyTorch model"""
    try:
        logger.warning("Custom PyTorch model extraction functionality needs implementation")
        return []
        
    except Exception as e:
        logger.error(f"PyTorch entity extraction failed: {e}")
        return []

def get_ner_model_info(model_dir: str = "transformer_ner_model") -> Dict:
    """Get NER model information"""
    info = {
        "model_dir": model_dir,
        "model_loaded": _ner_model is not None,
        "model_type": "unknown"
    }
    
    if os.path.exists(model_dir):
        if os.path.exists(os.path.join(model_dir, "config.json")):
            info["model_type"] = "transformers"
            try:
                with open(os.path.join(model_dir, "config.json"), 'r') as f:
                    config = json.load(f)
                info["model_name"] = config.get("_name_or_path", "unknown")
                info["num_labels"] = config.get("num_labels", "unknown")
            except:
                pass
                
        elif os.path.exists(os.path.join(model_dir, "meta.json")):
            info["model_type"] = "spacy"
            try:
                with open(os.path.join(model_dir, "meta.json"), 'r') as f:
                    meta = json.load(f)
                info["model_name"] = meta.get("name", "unknown")
                info["version"] = meta.get("version", "unknown")
            except:
                pass
                
        elif os.path.exists(os.path.join(model_dir, "model.pth")):
            info["model_type"] = "pytorch"
    
    return info

def test_ner_model(model_dir: str = "transformer_ner_model"):
    """Test NER model"""
    print("Testing NER model...")
    
    model = load_ner_model(model_dir)
    if not model:
        print("NER model loading failed")
        return
    
    info = get_ner_model_info(model_dir)
    print(f"Model information:")
    print(f"  Type: {info['model_type']}")
    print(f"  Directory: {info['model_dir']}")
    if 'model_name' in info:
        print(f"  Name: {info['model_name']}")
    
    test_texts = [
        "Apple Inc. is based in Cupertino, California and was founded by Steve Jobs.",
        "The research was conducted at MIT in collaboration with Harvard University.",
        "This project uses machine learning algorithms developed by Google and Microsoft.",
        "Dr. John Smith from Stanford University published this paper in Nature journal."
    ]
    
    print(f"\nTesting entity extraction:")
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        entities = model.predict(text)
        
        if entities:
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['label']}) [confidence: {entity['confidence']:.3f}]")
        else:
            print("  No entities found")

if __name__ == "__main__":
    print("NER Model Loader")
    print("=" * 50)
    
    model_dir = "transformer_ner_model"
    
    if os.path.exists(model_dir):
        print(f"Found NER model directory: {model_dir}")
        
        info = get_ner_model_info(model_dir)
        print(f"Model type: {info['model_type']}")
        
        test_ner_model(model_dir)
    else:
        print(f"NER model directory does not exist: {model_dir}")
        print("\nPlease ensure NER model files are copied to transformer_ner_model directory")
        print("Supported model types:")
        print("1. Transformers model (requires config.json)")
        print("2. spaCy model (requires meta.json)")
        print("3. PyTorch model (requires model.pth)")
        
        print("\nUsage:")
        print("# Load model")
        print("model = load_ner_model('transformer_ner_model')")
        print("\n# Extract entities")
        print("entities = model.predict('Your text here')")