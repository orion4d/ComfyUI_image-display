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
                # -- OPTIONS DE SAUVEGARDE (INCHANGÉES) --
                "save_image": ("BOOLEAN", {"default": False}),
                "save_path": ("STRING", {"multiline": False, "default": ""}),
                "filename": ("STRING", {"multiline": False, "default": "DisplayWithMask"}),
                "add_datetime": ("BOOLEAN", {"default": True}),
            }
        }

    # --- MODIFICATION 1 : AJOUT DE LA SORTIE MASK ---
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("processed_image", "mask_out",)
    # --- FIN DE LA MODIFICATION 1 ---
    
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

        # La logique de sauvegarde reste inchangée
        if save_image:
            output_dir = folder_paths.get_output_directory()
            if save_path and save_path.strip():
                full_save_path = os.path.join(output_dir, save_path)
            else:
                full_save_path = output_dir
            os.makedirs(full_save_path, exist_ok=True)
            base_filename = filename.strip() if filename and filename.strip() else "DisplayWithMask"
            if add_datetime:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                final_filename = f"{base_filename}_{timestamp}.png"
            else:
                final_filename = f"{base_filename}.png"
            full_path = os.path.join(full_save_path, final_filename)
            img_pil.save(full_path, "PNG")
            print(f"[DisplayImageWithMask] Image sauvegardée dans : {full_path}")

        # La logique de prévisualisation reste inchangée
        previews = []
        preview_filename = f"preview_{uuid.uuid4()}.png"
        temp_path = os.path.join(folder_paths.get_temp_directory(), preview_filename)
        img_pil.save(temp_path, "PNG")
        previews.append({"filename": preview_filename, "subfolder": "", "type": "temp"})

        # --- MODIFICATION 2 : SÉPARATION DE L'IMAGE ET DU MASQUE POUR LES SORTIES ---
        
        # 1. Séparer l'image RGB et le canal Alpha (masque)
        image_rgb = img_pil.convert("RGB")
        mask_alpha = img_pil.getchannel('A')

        # 2. Convertir l'image RGB en tenseur IMAGE pour la première sortie
        image_out_tensor = np.array(image_rgb).astype(np.float32) / 255.0
        image_out_tensor = torch.from_numpy(image_out_tensor).unsqueeze(0)

        # 3. Convertir le canal Alpha en tenseur MASK pour la deuxième sortie
        mask_out_tensor = np.array(mask_alpha).astype(np.float32) / 255.0
        mask_out_tensor = torch.from_numpy(mask_out_tensor)

        # 4. Retourner la prévisualisation et les deux tenseurs dans "result"
        return { 
            "ui": {"images": previews}, 
            "result": (image_out_tensor, mask_out_tensor,) 
        }
        # --- FIN DE LA MODIFICATION 2 ---

NODE_CLASS_MAPPINGS = { "DisplayImageWithMask": DisplayImageWithMaskNode }
NODE_DISPLAY_NAME_MAPPINGS = { "DisplayImageWithMask": "🖼️ Display Image with Mask" }