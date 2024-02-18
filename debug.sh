python -m debugpy --wait-for-client --listen 5678 -m threefiner.cli mvdream --mesh "data/owl.ply" --prompt "an owl, 3d asset" --text_dir --front_dir='-y' --outdir . --save "owl-sd.glb"


# $HOME/git-repo/H3DGEN/dumps/boots-0-wobkg.ply
python -m debugpy --wait-for-client --listen 5678 -m threefiner.cli mvdream --mesh "$HOME/git-repo/H3DGEN/dumps/boots-0-wobkg.ply" --prompt "boots, 3d asset" --text_dir --front_dir='-y' --outdir . --save "boots-sd.glb" --gui
