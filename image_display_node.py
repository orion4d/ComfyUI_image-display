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
            # Handle both IMAGE (B, H, W, C) and MASK (B, H, W) inputs gracefully
            tensor = image_tensor[0]
            if len(tensor.shape) == 2: # It's a MASK (H, W)
                 np_hw = (tensor.cpu().numpy() * 255).astype(np.uint8)
                 return Image.fromarray(np_hw, mode="L")
            elif len(tensor.shape) == 3: # It's an IMAGE (H, W, C)
                np_hwc = (tensor.cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_hwc)
        
        return None

    def process(self, image=None, image_url="", mask=None, mask_url="", invert_mask=False, 
                save_image=False, save_path="", filename="DisplayWithMask", add_datetime=True):
        
        img_pil = self._load_pil_image(image_tensor=image, image_url=image_url)

        if img_pil is None:
            # Create a default transparent image if no input is provided, 
            # ensuring we always have an alpha channel to extract later.
            img_pil = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            print("[DisplayImageWithMask] [Avertissement] Aucune source d'image principale valide n'a été fournie. Utilisation d'une image transparente par défaut.")

        # Ensure the main image is RGBA so it has an alpha channel
        img_pil = img_pil.convert("RGBA")
        
        source_mask_pil = self._load_pil_image(image_tensor=mask, image_url=mask_url)
        
        if source_mask_pil is not None:
            source_mask_pil = source_mask_pil.convert("L")
            if source_mask_pil.size != img_pil.size:
                # Resize and center the mask if sizes differ
                final_mask_pil = Image.new("L", img_pil.size, 0)
                paste_x = (img_pil.width - source_mask_pil.width) // 2
                paste_y = (img_pil.height - source_mask_pil.height) // 2
                
                # Calculate cropping if mask is larger than image
                crop_x1 = max(0, -paste_x)
                crop_y1 = max(0, -paste_y)
                crop_x2 = min(source_mask_pil.width, img_pil.width - paste_x)
                crop_y2 = min(source_mask_pil.height, img_pil.height - paste_y)

                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                    cropped_mask = source_mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    paste_pos_x = max(0, paste_x)
                    paste_pos_y = max(0, paste_y)
                    final_mask_pil.paste(cropped_mask, (paste_pos_x, paste_pos_y))
            else:
                final_mask_pil = source_mask_pil
            
            if invert_mask:
                final_mask_pil = ImageOps.invert(final_mask_pil)
            
            # Apply the processed mask to the image's alpha channel
            img_pil.putalpha(final_mask_pil)

        # La logique de sauvegarde reste inchangée
        if save_image:
            output_dir = folder_paths.get_output_directory()
            if save_path and save_path.strip():
                # Allow absolute paths or paths relative to the output directory
                if os.path.isabs(save_path):
                     full_save_path = save_path
                else:
                    full_save_path = os.path.join(output_dir, save_path)
            else:
                full_save_path = output_dir
            
            os.makedirs(full_save_path, exist_ok=True)
            base_filename = filename.strip() if filename and filename.strip() else "DisplayWithMask"
            
            # Handle potential conflicts if add_datetime is false
            final_filename = f"{base_filename}.png"
            if add_datetime:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                final_filename = f"{base_filename}_{timestamp}.png"
            
            full_path = os.path.join(full_save_path, final_filename)
            
            # Simple conflict resolution if not using datetime
            if not add_datetime:
                counter = 1
                while os.path.exists(full_path):
                    final_filename = f"{base_filename}_{counter}.png"
                    full_path = os.path.join(full_save_path, final_filename)
                    counter += 1

            try:
                img_pil.save(full_path, "PNG")
                print(f"[DisplayImageWithMask] Image sauvegardée dans : {full_path}")
            except Exception as e:
                print(f"[DisplayImageWithMask] Erreur lors de la sauvegarde de l'image : {e}")

        # La logique de prévisualisation reste inchangée
        previews = []
        # Use a unique filename for the preview to avoid caching issues
        preview_filename = f"preview_{uuid.uuid4()}.png"
        temp_path = os.path.join(folder_paths.get_temp_directory(), preview_filename)
        
        # Save preview as PNG to preserve transparency in the UI
        img_pil.save(temp_path, "PNG")
        
        # ComfyUI expects the relative path from the temp directory
        previews.append({"filename": os.path.basename(temp_path), "subfolder": os.path.relpath(os.path.dirname(temp_path), folder_paths.get_temp_directory()), "type": "temp"})

        # --- MODIFICATION 2 : SÉPARATION DE L'IMAGE ET DU MASQUE POUR LES SORTIES ---
        
        # 1. Séparer l'image RGB et le canal Alpha (masque)
        # We extract RGB for the image output, discarding the alpha we just applied.
        image_rgb = img_pil.convert("RGB") 
        # We extract the Alpha channel (which is the mask we applied or the original alpha)
        mask_alpha = img_pil.getchannel('A')

        # 2. Convertir l'image RGB en tenseur IMAGE pour la première sortie (B, H, W, C)
        image_out_tensor = np.array(image_rgb).astype(np.float32) / 255.0
        image_out_tensor = torch.from_numpy(image_out_tensor).unsqueeze(0)

        # 3. Convertir le canal Alpha en tenseur MASK pour la deuxième sortie (B, H, W)
        mask_out_tensor = np.array(mask_alpha).astype(np.float32) / 255.0
        mask_out_tensor = torch.from_numpy(mask_out_tensor)
        
        # !!! CORRECTION ICI !!! 
        # Ajouter la dimension batch (unsqueeze(0)) pour correspondre au format MASK (B, H, W)
        mask_out_tensor = mask_out_tensor.unsqueeze(0)

        # 4. Retourner la prévisualisation et les deux tenseurs dans "result"
        return { 
            "ui": {"images": previews}, 
            "result": (image_out_tensor, mask_out_tensor,) 
        }
        # --- FIN DE LA MODIFICATION 2 ---

NODE_CLASS_MAPPINGS = { "DisplayImageWithMask": DisplayImageWithMaskNode }
NODE_DISPLAY_NAME_MAPPINGS = { "DisplayImageWithMask": "🖼️ Display Image with Mask" }