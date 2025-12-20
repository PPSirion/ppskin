import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.glfw import GlfwRenderer
import colorsys

# Parametri del piano e dell'onda
plane_size = 50
window_size = (800, 600)
# Parametri di simulazione
elasticity = 1.0  # Aumentato da 0.3 a 0.5 per maggiore propagazione
damping = 0.3     # Leggermente ridotto per meno smorzamento
# timestep di default: ~60Hz - ridotto rispetto al valore precedente per stabilità della simulazione
delta_time = 0.3
force = 10.0       # Forza applicata al clic del mouse (aumentata da 3.0)
base_mass = 1.0
# Altezza di riposo della griglia rispetto alla base (1 cm)
plane_rest_height = 1.0  # 1 cm sopra la base (unità del mondo)

# Piccolo termine di gravità (scalato): mantiene la pelle a contatto con la base
# gravity = -9.81 * 0.001

# Limite sulla velocità per evitare esplosioni numeriche
max_velocity = 5.0

# Inizializza l'altezza usando la quota di riposo
plane_height = np.full((plane_size, plane_size), plane_rest_height)
plane_velocity = np.zeros((plane_size, plane_size))
plane_acceleration = np.zeros((plane_size, plane_size))
plane_mass = np.ones((plane_size, plane_size))

# PID controller per stabilizzazione globale (toggleable)
pid_enabled = False
pid_kp = 2.0
pid_ki = 0.5
pid_kd = 0.1
pid_integral = 0.0
pid_prev_error = 0.0
pid_output_min = 0.0
pid_output_max = 1.0
pid_target_speed = 0.02  # velocità media target (m/s)
pid_last_output = 0.0

def pid_step(avg_speed, dt):
    """Semplice PID che prende l'errore sulla velocità media del sistema
    e restituisce un fattore di smorzamento aggiuntivo [0,1]."""
    global pid_integral, pid_prev_error, pid_last_output
    error = avg_speed - pid_target_speed
    pid_integral += error * dt
    derivative = (error - pid_prev_error) / dt if dt > 0 else 0.0
    out = pid_kp * error + pid_ki * pid_integral + pid_kd * derivative
    pid_prev_error = error
    # clamp output
    pid_last_output = max(pid_output_min, min(pid_output_max, out))
    return pid_last_output

# Inizializza l'altezza usando la quota di riposo
plane_height = np.full((plane_size, plane_size), plane_rest_height)
plane_velocity = np.zeros((plane_size, plane_size))
plane_acceleration = np.zeros((plane_size, plane_size))
plane_mass = np.ones((plane_size, plane_size))

# Controlli della telecamera
class Camera:
    def __init__(self, angle_x=-60, angle_y=20, zoom=-25):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.zoom = zoom
        self.last_pos = None

    def update_rotation(self, x, y):
        if self.last_pos is not None:
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            self.angle_y += dx * 0.2
            self.angle_x += dy * 0.2
        self.last_pos = (x, y)

    def reset_last_pos(self):
        self.last_pos = None

camera = Camera()

mouse_dragging = False
mouse_weight_active = False

# Aumentato il size della mini griglia
force_box_size = (350, 350)  # Aumentato da 220 a 350 (larghezza, altezza in pixel)
force_box_margin = 10

# Variabili per la mappatura del riquadro piccolo
force_box_params = {
    'child_x': 0,
    'child_y': 0,
    'cell_w': 0,
    'cell_h': 0,
    'start_y': 0,
    'padding': 0
}

def get_force_box_rect():
    # position the box at bottom-right with a small margin
    bw, bh = force_box_size
    bx = window_size[0] - bw - force_box_margin
    # move the box further up from the bottom
    force_box_vertical_offset = 100  # Ridotto da 150 a 100 per posizionare meglio la griglia più grande
    by = window_size[1] - bh - force_box_margin - force_box_vertical_offset
    if bx < 0:
        bx = 0
    if by < 0:
        by = 0
    return bx, by, bw, bh

def get_hex_offset(x):
    """Restituisce l'offset per la griglia esagonale."""
    return (x % 2) * 0.5

def compute_spring_forces(height_grid, velocity_grid):
    """Calcola la forza netta su ogni cella:
    - somma dei contributi Hooke + viscous damper verso ognuno dei 6 vicini (non media)
    - aggiunge la molla+damper verso la base (posizione di riposo = plane_rest_height)
    - aggiunge un piccolo termine di gravità
    Restituisce un array con la somma delle forze per cella (positivo = spinge verso su)."""
    net_force = np.zeros((plane_size, plane_size))

    # Usa i parametri globali: elasticity come costante della molla (k), damping come coefficiente del damper (c)
    k = elasticity
    c = damping

    for x in range(plane_size):
        for y in range(plane_size):
            # definisci i 6 vicini come prima (parita' della colonna)
            if x % 2 == 0:  # riga/parita' pari
                neighbors = [
                    (x-1, y),   # sinistra
                    (x+1, y),   # destra
                    (x, y-1),   # basso
                    (x, y+1),   # alto
                    (x-1, y+1), # alto-sinistra
                    (x+1, y+1)  # alto-destra
                ]
            else:  # riga dispari
                neighbors = [
                    (x-1, y),   # sinistra
                    (x+1, y),   # destra
                    (x, y-1),   # basso
                    (x, y+1),   # alto
                    (x-1, y-1), # basso-sinistra
                    (x+1, y-1)  # basso-destra
                ]

            hz = height_grid[x, y]
            hv = velocity_grid[x, y]

            # somma contributi da ogni vicino (Hooke + viscous damper sui tassi relativi)
            f_sum = 0.0
            for nx, ny in neighbors:
                if 0 <= nx < plane_size and 0 <= ny < plane_size:
                    neighbor_z = height_grid[nx, ny]
                    neighbor_v = velocity_grid[nx, ny]
                    dz = neighbor_z - hz         # estensione della molla (positivo se vicino piu' alto)
                    dv = neighbor_v - hv         # velocita' relativa
                    # forza diretta dalla molla verso il vicino
                    f_sum += k * dz + c * dv

            # Connessione alla base fissa: usiamo plane_rest_height come posizione di riposo
            # La forza della molla verso la base annulla quando hz == plane_rest_height
            base_force = k * (plane_rest_height - hz) + c * (0.0 - hv)

            # Piccolo termine di gravità (sommato come forza netta)
            # gravity_force = gravity * max(1.0, plane_mass[x, y])

            # Somma delle forze (positive = spinge verso su)
            # net_force[x, y] = f_sum + base_force + gravity_force
            net_force[x, y] = f_sum + base_force

    return net_force


def update_plane():
    global plane_height, plane_velocity, plane_acceleration

    # Applica il peso se il mouse è premuto
    if mouse_weight_active:
        apply_force_at_cursor(glfw.get_current_context())

    # Calcola le forze di molla e smorzamento tra vicini (senza usare il laplaciano)
    net_force = compute_spring_forces(plane_height, plane_velocity)

    # acceleration = net_force / mass (nessun altro termine esterno; solo interazione tra elementi)
    safe_mass = np.maximum(plane_mass, 1e-6)
    plane_acceleration = net_force / safe_mass

    # integrazione esplicita semi-implicita (velocita' aggiornata, poi posizione)
    plane_velocity += plane_acceleration * delta_time

    # Clamp della velocita' per prevenire esplosioni numeriche
    np.clip(plane_velocity, -max_velocity, max_velocity, out=plane_velocity)

    plane_height += plane_velocity * delta_time

    # PID: stabilizzazione globale (riduce la velocità media se attivo)
    if pid_enabled:
        avg_speed = float(np.mean(np.abs(plane_velocity)))
        extra_damp = pid_step(avg_speed, delta_time)
        if extra_damp > 0.0:
            # applichiamo lo smorzamento come fattore moltiplicativo alle velocità
            plane_velocity *= max(0.0, 1.0 - extra_damp)

    # PREVENZIONE SEMPLICE DELLA PENETRAZIONE DELLA BASE:
    # Se un nodo scende sotto z=0 lo riportiamo a 0 e applichiamo una piccola restituzione
    below = plane_height < 0.0
    if np.any(below):
        restitution = 0.15
        plane_height[below] = 0.0
        plane_velocity[below] = -plane_velocity[below] * restitution

    # BORDI LIBERI: non ancoriamo i nodi del perimetro; permettiamo alla pelle di muoversi liberamente ai bordi.
    # Per ridurre artefatti numerici applichiamo un leggero smorzamento alle velocità di bordo invece di fissarle.
    edge_damping = 0.995
    plane_velocity[0, :] *= edge_damping
    plane_velocity[-1, :] *= edge_damping
    plane_velocity[:, 0] *= edge_damping
    plane_velocity[:, -1] *= edge_damping

    # Manteniamo l'accelerazione coerente (può rimanere calcolata normalmente)
    # ...existing code...

def reset_simulation():
    global plane_height, plane_velocity, plane_acceleration
    plane_height.fill(plane_rest_height)
    plane_velocity.fill(0)
    plane_acceleration.fill(0)

def mouse_button_callback(window, button, action, mods):
    global mouse_dragging, mouse_weight_active
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            # Ottieni coordinate del mouse
            cx, cy = glfw.get_cursor_pos(window)
            # Converti in coordinate della griglia esagonale
            grid_x, grid_y = force_box_mouse_to_grid(cx, cy)
            if grid_x is not None and grid_y is not None:
                mouse_dragging = True
                mouse_weight_active = True
                apply_force_at_cursor(window, from_grid=True, grid_pos=(grid_x, grid_y))
            else:
                mouse_dragging = False
                mouse_weight_active = False
        elif action == glfw.RELEASE:
            mouse_dragging = False
            mouse_weight_active = False

def cursor_position_callback(window, xpos, ypos):
    global mouse_dragging
    if mouse_dragging and mouse_weight_active:
        # Ottieni coordinate del mouse e converti
        grid_x, grid_y = force_box_mouse_to_grid(xpos, ypos)
        if grid_x is not None and grid_y is not None:
            apply_force_at_cursor(window, from_grid=True, grid_pos=(grid_x, grid_y))

def apply_force_at_cursor(window, from_grid=False, grid_pos=None):
    """Apply force ONLY to the single clicked hex element (not neighbors)."""
    global plane_velocity
    if from_grid and grid_pos is not None:
        grid_x, grid_y = grid_pos
    else:
        # Fallback: usa il vecchio metodo se necessario
        cx, cy = glfw.get_cursor_pos(window)
        grid_x, grid_y = force_box_mouse_to_grid(cx, cy)
        if grid_x is None or grid_y is None:
            return
    
    # Apply force ONLY to the clicked cell (not neighbors)
    if 0 <= grid_x < plane_size and 0 <= grid_y < plane_size:
        m = max(1e-6, plane_mass[grid_x, grid_y])
        # Scale the applied impulse by delta_time so repeated clicks/drags don't create huge velocities
        plane_velocity[grid_x, grid_y] -= ((force * 1.2) / m) * delta_time

def force_box_mouse_to_grid(mx, my):
    """Convert mouse coordinates (relative to window) to grid coordinates (ix, iy) in the force box.
    Returns (ix, iy) or (None, None) if outside any cell."""
    global force_box_params
    
    # Estrai parametri
    child_x = force_box_params['child_x']
    child_y = force_box_params['child_y']
    cell_w = force_box_params['cell_w']
    cell_h = force_box_params['cell_h']
    start_y = force_box_params['start_y']
    padding = force_box_params['padding']
    
    # Verifica se (mx, my) è dentro l'area del child (con padding)
    if not (child_x <= mx < child_x + plane_size * cell_w + padding and
            child_y <= my < child_y + (plane_size + 0.5) * cell_h + padding):
        return None, None
    
    # Converti in coordinate relative all'area di disegno (senza padding)
    rel_x = mx - child_x
    rel_y = my - child_y
    
    # Calcola l'indice di colonna ix
    ix = int(rel_x / cell_w)
    if ix < 0 or ix >= plane_size:
        return None, None
    
    # Calcola l'offset verticale per questa colonna (0 o 0.5 in unità cella)
    offset = get_hex_offset(ix) * 0.5
    
    # Calcola l'indice di riga iy
    # Prima converti in coordinate "griglia" normalizzate
    grid_y = (rel_y - offset * cell_h) / cell_h
    
    # La griglia è invertita verticalmente nel display
    iy = int(grid_y)
    
    # Verifica che sia dentro i limiti
    if iy < 0 or iy >= plane_size:
        return None, None
    
    # Verifica finale che il punto sia dentro l'esagono (approssimato come rettangolo)
    # Per semplicità, controlliamo solo i bordi del rettangolo
    cell_x0 = ix * cell_w
    cell_y0 = iy * cell_h + offset * cell_h
    
    if (cell_x0 <= rel_x < cell_x0 + cell_w and 
        cell_y0 <= rel_y < cell_y0 + cell_h):
        return ix, plane_size - 1 - iy  # Inverti verticalmente per matchare la griglia 3D
    
    return None, None

def setup_lighting_and_color():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, np.array([1.0, 1.0, 1.0, 0.0]))
    glPointSize(3)

def get_color(z):
    # Colore stabile per ogni altezza: mappa lineare da blu (basso) a rosso (alto)
    max_abs = 2.0  # imposta la gamma di altezze attesa
    z_norm = max(-max_abs, min(max_abs, z))
    if abs(z_norm) < 0.01:
        return (1.0, 1.0, 1.0)
    # Hue da 0.6 (blu) a 0.0 (rosso)
    h = 0.6 - 0.6 * ((z_norm + max_abs) / (2 * max_abs))
    s = min(1.0, abs(z_norm) / max_abs)
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def color_to_u32(r, g, b, a=1.0):
    r_i = int(max(0, min(255, int(r * 255))))
    g_i = int(max(0, min(255, int(g * 255))))
    b_i = int(max(0, min(255, int(b * 255))))
    a_i = int(max(0, min(255, int(a * 255))))
    return (a_i << 24) | (b_i << 16) | (g_i << 8) | r_i

def draw_plane():
    glBegin(GL_POINTS)
    for x in range(plane_size):
        for y in range(plane_size):
            z = plane_height[x, y]
            glColor3f(*get_color(z))
            glVertex3f(x, y + get_hex_offset(x), z)
    glEnd()

def draw_empty_hexagons():
    """Disegna il wireframe esagonale che mostra le 6 connessioni."""
    glBegin(GL_LINES)
    for x in range(plane_size):
        for y in range(plane_size):
            z = plane_height[x, y]
            glColor3f(*get_color(z))
            
            # Definisci i 6 vertici dell'esagono
            vertices = []
            for i in range(6):
                angle = i * np.pi / 3
                vx = x + 0.5 * np.cos(angle)
                vy = y + 0.5 * np.sin(angle) + get_hex_offset(x)
                vz = z
                vertices.append((vx, vy, vz))
            
            # Disegna linee che connettono vertici consecutivi
            for i in range(6):
                x1, y1, z1 = vertices[i]
                x2, y2, z2 = vertices[(i + 1) % 6]
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
    glEnd()

# Aggiunta: disegna la base a z=0 (piano semi-trasparente + griglia)
def draw_base():
    """Disegna un piano di base a z=0 coprendo l'area della griglia."""
    # Disattiva temporaneamente l'illuminazione per colore piatto e usa blending per trasparenza
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Piano riempito leggermente scuro e trasparente
    glColor4f(0.18, 0.18, 0.18, 0.6)
    glBegin(GL_QUADS)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(float(plane_size), 0.0, 0.0)
    glVertex3f(float(plane_size), float(plane_size), 0.0)
    glVertex3f(0.0, float(plane_size), 0.0)
    glEnd()

    # Linee della griglia sopra il piano
    glColor4f(0.0, 0.0, 0.0, 0.35)
    glBegin(GL_LINES)
    for i in range(plane_size + 1):
        # linee parallele asse Y
        glVertex3f(float(i), 0.0, 0.0)
        glVertex3f(float(i), float(plane_size), 0.0)
        # linee parallele asse X
        glVertex3f(0.0, float(i), 0.0)
        glVertex3f(float(plane_size), float(i), 0.0)
    glEnd()

    # restore blending / lighting
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)

def cleanup():
    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)
    glDisable(GL_DEPTH_TEST)

def main():
    global elasticity, damping, delta_time, force, window_size, max_velocity, base_mass, pid_enabled, pid_kp, pid_ki, pid_kd, pid_target_speed, pid_integral, pid_prev_error, pid_last_output

    if not glfw.init():
        return

    # Ottieni il monitor primario e la sua risoluzione
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    fullscreen_size = (mode.size.width, mode.size.height)

    # Imposta la finestra come non decorata (senza bordi)
    window = glfw.create_window(fullscreen_size[0], fullscreen_size[1], "3D Simulation", None, None)
    if not window:
        glfw.terminate()
        return

    # Aggiorna window_size per la simulazione e la conversione coordinate mouse
    window_size = fullscreen_size

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glEnable(GL_DEPTH_TEST)

    imgui.create_context()
    impl = GlfwRenderer(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        # Finestra ImGui per i parametri
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(320, 260)  # aumentata per i controlli PID
        flags = imgui.WINDOW_NO_COLLAPSE
        imgui.begin("Parameters", True, flags)
        imgui.text("Propagation Settings:")
        _, elasticity = imgui.slider_float("k", elasticity, 0.1, 2.0)
        _, damping = imgui.slider_float("c", damping, 0.01, 1.0)
        _, delta_time = imgui.slider_float("dt", delta_time, 0.001, 0.5)
        _, force = imgui.slider_float("F", force, 0.1, 15.0)
        global base_mass
        _, base_mass = imgui.slider_float("m", base_mass, 0.1, 10.0)
        _, max_velocity = imgui.slider_float("v_max", max_velocity, 0.1, 50.0)

        imgui.separator()
        imgui.text("PID Stabilization (global):")
        clicked, pid_enabled = imgui.checkbox("Enable PID", pid_enabled)
        imgui.same_line()
        if imgui.button("Reset PID"):
            pid_integral = 0.0
            pid_prev_error = 0.0
            pid_last_output = 0.0

        # PID gains and target
        _, pid_kp = imgui.slider_float("PID Kp", pid_kp, 0.0, 10.0)
        _, pid_ki = imgui.slider_float("PID Ki", pid_ki, 0.0, 5.0)
        _, pid_kd = imgui.slider_float("PID Kd", pid_kd, 0.0, 5.0)
        _, pid_target_speed = imgui.slider_float("PID target speed", pid_target_speed, 0.0, 0.5)

        # Visualizza output PID corrente
        imgui.text("PID output: {:.4f}".format(pid_last_output))

        if imgui.button("Reset Simulation"):
            reset_simulation()
        imgui.end()

        # Controlli della telecamera (ruota con il tasto destro del mouse)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            camera.update_rotation(x, y)
        else:
            camera.reset_last_pos()

        update_plane()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(40, (window_size[0] / window_size[1]), 0.1, 100.0)
        glTranslatef(-25, -10, camera.zoom)
        glRotatef(camera.angle_x, 1, 0, 0)
        glRotatef(camera.angle_y, 0, 0, 1)
        glTranslatef(8, 0, 0)

        setup_lighting_and_color()
        draw_base()
        draw_plane()
        draw_empty_hexagons()

        # Draw the force box as a miniature grid with hexagonal layout
        bx, by, bw, bh = get_force_box_rect()
        imgui.set_next_window_position(bx, by)
        imgui.set_next_window_size(bw, bh)
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR
        imgui.begin("Force Box", False, flags)
        
        # remove internal window padding so content fits exactly
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        padding = 4  # Aumentato il padding per rendere le celle più visibili
        
        # use available content region to size the grid so no scrolling is needed
        avail_w, avail_h = imgui.get_content_region_available()
        child_w = max(0, avail_w - padding * 2)
        child_h = max(0, avail_h - padding * 2)
        
        child_flags = imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE
        imgui.begin_child("force_grid_child", child_w, child_h, border=False, flags=child_flags)
        
        # draw miniature grid using screen-space origin from cursor
        draw_list = imgui.get_window_draw_list()
        origin_x, origin_y = imgui.get_cursor_screen_pos()
        
        # small inner offset
        child_x = origin_x + padding
        child_y = origin_y + padding
        
        # Calcola la dimensione della cella per adattare la griglia esagonale
        # L'altezza totale necessaria è (plane_size + 0.5) * cell_h a causa dell'offset
        cell_size = min(child_w / plane_size, child_h / (plane_size + 0.5))
        cell_w = cell_size
        cell_h = cell_size
        
        # Calcola l'altezza totale della griglia con offset
        total_grid_height = (plane_size + 0.5) * cell_h
        start_y = child_y + (child_h - total_grid_height) / 2
        
        # Salva i parametri per la mappatura del mouse
        force_box_params['child_x'] = child_x
        force_box_params['child_y'] = child_y
        force_box_params['cell_w'] = cell_w
        force_box_params['cell_h'] = cell_h
        force_box_params['start_y'] = start_y
        force_box_params['padding'] = padding
        
        # Disegna la griglia esagonale
        for ix in range(plane_size):
            for iy in range(plane_size):
                # Calcola la posizione con offset esagonale
                x0 = child_x + ix * cell_w
                offset = get_hex_offset(ix) * 0.5  # 0 o 0.5
                y0 = start_y + (plane_size - 1 - iy) * cell_h + offset * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                
                # Ottieni il colore in base all'altezza
                z = plane_height[ix, iy]
                r, g, b = get_color(z)
                col = color_to_u32(r, g, b, 1.0)
                
                # Disegna il rettangolo (cella esagonale approssimata)
                draw_list.add_rect_filled(x0, y0, x1, y1, col)
                
                # Bordo sottile - aumentata l'opacità per visibilità
                border_col = color_to_u32(0.0, 0.0, 0.0, 0.25)
                draw_list.add_rect(x0, y0, x1, y1, border_col)
        
        imgui.end_child()
        imgui.pop_style_var()
        imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    cleanup()
    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
    cleanup()
    glfw.terminate()