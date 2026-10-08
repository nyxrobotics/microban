# Software Installation & Deployment

This guide takes you from the pre-built Microban SD-card image to a robot you can drive
from your computer. The image already contains the operating system and the Microban
code; you will flash it, connect to the robot over the network, and run it. For
day-to-day operation and development afterwards, see the [Usage Guide](usage.md).

## Step 1: Download and Flash the Image

Download `microban.img.xz` from the
[latest image release](https://github.com/MarcDcls/microban/releases/tag/image.v2)
([direct link](https://github.com/MarcDcls/microban/releases/download/image.v2/microban.img.xz)).
There is no need to decompress it; Raspberry Pi Imager reads the compressed `.xz`
directly.

Then use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to write
`microban.img.xz` onto a micro-SD card. The card should be 16 GB or larger, and I
advise using a new card, as corrupted cards can cause issues.

1. Insert the micro-SD card into your computer.
2. Open Raspberry Pi Imager. It is distributed as an AppImage, so make it executable
   and run it as root to let it write to the SD card:
   ```bash
   chmod +x Downloads/imager_*.AppImage
   sudo Downloads/imager_*.AppImage
   ```
3. **Choose Device** → *Raspberry Pi Zero 2 W*.
4. **Choose OS** → scroll to the bottom and select *Use custom*, then pick the
   `microban.img.xz` file.
5. **Choose Storage** → select your micro-SD card.
6. Click **Next**. When asked *"Would you like to apply OS customisation settings?"*,
   choose **No** — the image is already configured (the Wi-Fi is set up in Step 2).
7. Click **Write** and wait for the flashing and verification to complete.

> Alternatively, on Linux you can flash from the command line. **Double-check the
> device name** (`lsblk`) — writing to the wrong disk will erase it:
> ```bash
> xzcat microban.img.xz | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
> ```

## Step 2: Headless Wi-Fi Configuration

Once the flashing process is complete, unplug and plug back the micro-SD card into your Ubuntu PC. Both the `bootfs` and `rootfs` partitions get mounted.

📶 CRITICAL: the network **must be 2.4 GHz**. The Raspberry Pi Zero 2 W has a 2.4 GHz-only
Wi-Fi chip and **cannot see or join a 5 GHz network** — it will silently never connect. If
you use a phone hotspot, force it to 2.4 GHz.

Wi-Fi is configured by writing a NetworkManager connection profile directly onto
`rootfs` (**not** by editing `network-config` — see why below). Generate a UUID and
write the profile:

```bash
UUID=$(uuidgen)
sudo tee /media/$USER/rootfs/etc/NetworkManager/system-connections/home.nmconnection >/dev/null <<EOF
[connection]
id=<YOUR_WIFI_NAME>
uuid=$UUID
type=wifi
autoconnect=true

[wifi]
mode=infrastructure
ssid=<YOUR_WIFI_NAME>

[wifi-security]
key-mgmt=wpa-psk
psk=<YOUR_WIFI_PASSWORD>

[ipv4]
method=auto

[ipv6]
method=auto
addr-gen-mode=default

[proxy]
EOF
sudo chown root:root /media/$USER/rootfs/etc/NetworkManager/system-connections/home.nmconnection
sudo chmod 600 /media/$USER/rootfs/etc/NetworkManager/system-connections/home.nmconnection
```

⚠️ CRITICAL: the file must be owned by `root:root` with `600` permissions, or
NetworkManager silently ignores it.

🛜 Replace both `<YOUR_WIFI_NAME>` and `<YOUR_WIFI_PASSWORD>` with your local network
credentials. To also reach the robot on a second network (e.g. a phone hotspot), repeat
the block above with a different file name (e.g. `hotspot.nmconnection`) — the robot
tries every profile it has and connects to whichever is in range.

Safely eject the card from your Ubuntu PC, insert it into the Raspberry Pi Zero 2W, and power it up.

> **Why not `network-config`?** That file still exists on `bootfs`, and it looks like the
> obvious place to set Wi-Fi — but on this image, cloud-init's Wi-Fi config never gets
> turned into an actual NetworkManager connection (a cloud-init module it depends on,
> `cc_netplan_nm_patch`, is missing from this image). It validates with no error and
> silently does nothing, so the robot never joins your network. Leave `network-config` at
> its placeholder values and use the `.nmconnection` method above instead.

> **Changing the Wi-Fi network later:** with the SD card back in your PC, add, replace, or
> delete `.nmconnection` files under `rootfs/etc/NetworkManager/system-connections/` the
> same way, then reboot the Pi. If you already have SSH access, you can instead run
> `sudo nmcli device wifi connect "<SSID>" password "<PASSWORD>"` directly on the Pi.

> **Country / regulatory domain:** this method does not set the Wi-Fi regulatory domain,
> so channels 12–13 stay unavailable. Most routers default to channel 1, 6, or 11, so this
> rarely matters; if you need those channels, run
> `sudo raspi-config nonint do_wifi_country <CC>` once you have SSH access.

## Step 3: First SSH Connection

Give the Raspberry Pi about 1 to 2 minutes on its very first boot. It will automatically resize the file system to use the full capacity of your SD card and then connect to your Wi-Fi network. Do not power it off during this process, as it may take a while and interrupting it could cause issues.

When the Pi is ready, you should be able to ping it from your computer using the command:

```bash
ping microban.local
```

You should see a response looking like this:
```
PING microban.local (192.168.XXX.XX) 56(84) bytes of data.
64 bytes from 192.168.XXX.XXX: icmp_seq=1 ttl=64 time=100 ms
...
```

The default credentials of this image are:
- Username: `user`
- Password: `password`

Open your terminal and connect over SSH:

```bash
ssh user@microban.local
```

> If `microban.local` cannot be found after 5 minutes, look up your local router's DHCP client list
> to find the IP address assigned to the Pi, and connect with `ssh user@<IP>`.

## Step 4: Personalize and Secure Your Robot

Once you are successfully connected via SSH, change the default password and,
optionally, customize the hostname.

### Change the password

Run the password tool and follow the prompts:
```bash
passwd
```

Apply the changes:
```bash
sudo reboot
```

### OPTIONAL - Change the username

The image ships with the user `user`. A logged-in account cannot be renamed, so create
a temporary admin account, rename from there, then remove it. Replace `NEW_USER` with
your choice.

From the default `user` session, create a temporary admin and set its password:
```bash
sudo useradd -m -s /bin/bash -G sudo tmpadmin
sudo passwd tmpadmin
```
Log out, reconnect as `tmpadmin`, then rename the original account:
```bash
ssh tmpadmin@<ROBOT_IP>
sudo usermod -l NEW_USER -d /home/NEW_USER -m user
sudo groupmod -n NEW_USER user      # rename its primary group too
```
Log out, reconnect as `NEW_USER`, and delete the temporary account:
```bash
ssh NEW_USER@<ROBOT_IP>
sudo userdel -r tmpadmin
```
The shutdown-without-password rule still works (it targets the `sudo` group, not a
name). Remember to update the `User` field in your `~/.ssh/config` (Step 5).

Apply the changes:
```bash
sudo reboot
```

### OPTIONAL - Change the hostname

Rename your robot so it no longer answers to `microban` on the network:
```bash
sudo hostnamectl set-hostname NEW_ROBOT_NAME
```
Then update the hosts file to match:
```bash
sudo nano /etc/hosts
```
Find the line containing `microban` (usually the second line) and replace it with your
`NEW_ROBOT_NAME`. Save and exit (Ctrl+O, Enter, Ctrl+X).

Apply the changes:
```bash
sudo reboot
```

## Step 5: Deploy the Software from Your Computer

The code already ships on the robot, but you drive and update it from your computer
using the provided `Makefile`, which syncs your local copy to the Pi over SSH and
installs the dependencies there.

1. **Install [uv](https://docs.astral.sh/uv/)** (the Python package manager) on your
   computer:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository** on your computer:
   ```bash
   git clone https://github.com/MarcDcls/microban.git
   cd microban
   ```

3. **Add SSH aliases** so the Makefile can reach the Pi. Add the following to your
   `~/.ssh/config`, replacing `user` if you changed it and `<IP_ADDRESS>` with the
   robot's IP on each network:
   ```
   # Main network
   Host microban
       HostName <IP_ADDRESS>
       User user

   # Secondary network / phone hotspot
   Host microban-ext
       HostName <IP_ADDRESS>
       User user
   ```
   To find the robot's IP on a given network ping it from your computer while connected to that network:
   ```bash
   ping microban.local
   ```

   > The Makefile targets default to `HOST=microban`. To operate the robot over the
   > secondary network, pass `HOST=microban-ext` (e.g. `make run HOST=microban-ext`),
   > or export it for the whole session: `export HOST=microban-ext`.

4. **Copy your SSH key** (recommended) so you are not prompted for a password on every
   command:
   ```bash
   ssh-keygen -t ed25519     # skip if you already have a key
   ssh-copy-id microban
   ```

5. **Push the code and install dependencies** on the Pi:
   ```bash
   make setup
   ```
   This rsyncs your local copy to the robot and runs `uv sync --frozen` there.

## Step 6: Run the Robot

Your Microban is ready! Place it on a stable surface (or hold it securely), then start
the control loop from your computer:

```bash
make run
```

On start the robot enables torque and ramps to its neutral pose, then runs the control
loop at 50 Hz attached to your terminal. Press `q` or run `make stop` to stop.

🎉 For the full set of controls — Makefile commands, keyboard and gamepad driving,
the available moves, and how to add your own — see the **[Usage Guide](usage.md)**.
