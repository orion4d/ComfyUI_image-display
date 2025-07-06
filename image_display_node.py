import os
import torch
import folder_paths
import numpy as np
from PIL import Image, ImageOps
import uuid
import io
from datetime import datetime # Importation nécessaire pour l'horodatage

# Forcer le support AVIF pour l'ensemble de ComfyUI au démarrage
try:
    import pillow_avif
    print("[DisplayImageWithMask] Support AVIF activé.")
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("[DisplayImageWithMask] La bibliothèque 'requests' n'est pas installée. Le chargement d'URL ne fonctionnera pas.")
    requests = None

class DisplayImageWithMaskNode:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "mask": ("IMAGE",),
                "mask_url": ("STRING", {"multiline": False, "default": ""}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                # -- NOUVELLES OPTIONS DE SAUVEGARDE --
                "save_image": ("BOOLEAN", {"default": False}),
                "save_path": ("STRING", {"multiline": False, "default": ""}),
                "filename": ("STRING", {"multiline": False, "default": "DisplayWithMask"}),
                "add_datetime": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process"
    CATEGORY = "utils/display"

    def _load_pil_image(self, image_tensor=None, image_url=""):
        if image_url and image_url.strip() and requests:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            except Exception as e:
                print(f"[DisplayImageWithMask] [Avertissement] Échec du téléchargement de l'URL {image_url}: {e}")
        
        if image_tensor is not None:
            tensor_hwc = image_tensor[0]
            np_hwc = (tensor_hwc.cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_hwc)
        
        return None

    def process(self, image=None, image_url="", mask=None, mask_url="", invert_mask=False, 
                save_image=False, save_path="", filename="DisplayWithMask", add_datetime=True):
        
        img_pil = self._load_pil_image(image_tensor=image, image_url=image_url)

        if img_pil is None:
            raise ValueError("[DisplayImageWithMask] Aucune source d'image principale valide n'a été fournie.")

        img_pil = img_pil.convert("RGBA")
        source_mask_pil = self._load_pil_image(image_tensor=mask, image_url=mask_url)
        
        if source_mask_pil is not None:
            source_mask_pil = source_mask_pil.convert("L")
            if source_mask_pil.size != img_pil.size:
                final_mask_pil = Image.new("L", img_pil.size, 0)
                paste_x = (img_pil.width - source_mask_pil.width) // 2
                paste_y = (img_pil.height - source_mask_pil.height) // 2
                final_mask_pil.paste(source_mask_pil, (paste_x, paste_y))
            else:
                final_mask_pil = source_mask_pil
            if invert_mask:
                final_mask_pil = ImageOps.invert(final_mask_pil)
            img_pil.putalpha(final_mask_pil)

        # Sauvegarde permanente si l'option est cochée
        if save_image:
            # -- NOUVELLE LOGIQUE DE SAUVEGARDE PERSONNALISÉE --
            output_dir = folder_paths.get_output_directory()
            
            # 1. Déterminer le dossier de destination
            if save_path and save_path.strip():
                # Si un chemin est fourni, on le joint au dossier de sortie principal
                full_save_path = os.path.join(output_dir, save_path)
            else:
                full_save_path = output_dir

            # 2. Créer le dossier s'il n'existe pas
            os.makedirs(full_save_path, exist_ok=True)
            
            # 3. Construire le nom du fichier
            base_filename = filename.strip() if filename and filename.strip() else "DisplayWithMask"
            
            if add_datetime:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                final_filename = f"{base_filename}_{timestamp}.png"
            else:
                final_filename = f"{base_filename}.png"

            # 4. Combiner pour obtenir le chemin complet et sauvegarder
            full_path = os.path.join(full_save_path, final_filename)
            img_pil.save(full_path, "PNG")
            print(f"[DisplayImageWithMask] Image sauvegardée dans : {full_path}")

        # Préparation de l'aperçu et de la sortie (inchangé)
        previews = []
        preview_filename = f"preview_{uuid.uuid4()}.png"
        temp_path = os.path.join(folder_paths.get_temp_directory(), preview_filename)
        img_pil.save(temp_path, "PNG")
        previews.append({"filename": preview_filename, "subfolder": "", "type": "temp"})

        img_out_np = np.array(img_pil).astype(np.float32) / 255.0
        img_out_tensor = torch.from_numpy(img_out_np).unsqueeze(0)

        return { "ui": {"images": previews}, "result": (img_out_tensor,) }

NODE_CLASS_MAPPINGS = { "DisplayImageWithMask": DisplayImageWithMaskNode }
NODE_DISPLAY_NAME_MAPPINGS = { "DisplayImageWithMask": "🖼️ Display Image with Mask" }