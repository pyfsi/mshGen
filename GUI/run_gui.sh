#!/bin/bash
# run_gui.sh: start the mshGen GUI (Panel) and open it in your browser.
#
#   GUI/run_gui.sh [port]          (default port 5006)
#
# The server listens on this machine only (127.0.0.1). In VS Code Remote, the port is forwarded
# automatically (see the PORTS tab); open http://localhost:<port>/app in your local browser.
# The login asks for your user name and a password, stored in ~/.config/mshgen_gui/users.json
# (created on first start, readable only by you), so other users logged in on the same machine
# cannot use your GUI. To change the password, edit or delete that file.
cd "$(dirname "$0")" || exit
PORT=${1:-5006}

type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
module load Anaconda3-python/2023.09-0
export MPLBACKEND=Agg

# Login file for Panel's basic authentication: JSON {"<user>": "<password>"}
PASSFILE="$HOME/.config/mshgen_gui/users.json"
if [ ! -s "$PASSFILE" ]; then
    mkdir -p "$(dirname "$PASSFILE")"
    chmod 700 "$(dirname "$PASSFILE")"
    ( umask 077; python -c "import json, os, secrets; print(json.dumps({os.environ['USER']: secrets.token_urlsafe(12)}))" > "$PASSFILE" )
fi
chmod 600 "$PASSFILE"
COOKIE_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

echo "mshGen GUI: http://localhost:$PORT/app"
echo "Login: user name and password from $PASSFILE:"
cat "$PASSFILE"
exec panel serve app.py --port "$PORT" --address 127.0.0.1 \
    --allow-websocket-origin "localhost:$PORT" --allow-websocket-origin "127.0.0.1:$PORT" \
    --basic-auth "$PASSFILE" --cookie-secret "$COOKIE_SECRET"
