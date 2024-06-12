import sqlite3

def add_employee(employee_id, name, department, position):
    conn = sqlite3.connect('employee_tracking.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO Employee (id, name, department, position)
    VALUES (?, ?, ?, ?)
    ''', (employee_id, name, department, position))

    conn.commit()
    conn.close()
    print(f"Added employee {name} with ID {employee_id}")

if __name__ == "__main__":
    add_employee("E001", "Sadia", "sewing", "worker")
    add_employee("E002", "Faria", "sewing", "worker")
    add_employee("E003", "Ahnaf", "sewing", "worker")
    add_employee("E004", "Kamee", "sewing", "worker")
    add_employee("E005", "Njia", "sewing", "worker")
    add_employee("E006", "Rifat", "sewing", "worker")
    add_employee("E007", "Nasir", "sewing", "worker")
    add_employee("E008", "Abir", "sewing", "worker")
    add_employee("E009", "Pushpa", "sewing", "worker")
    add_employee("E010", "Sargam", "sewing", "worker")
    add_employee("E011", "Rajib", "sewing", "worker")
    add_employee("E012", "Omor", "sewing", "worker")
    add_employee("E013", "Faruk", "sewing", "worker")
    add_employee("E014", "Abdullah", "sewing", "worker")
    add_employee("E015", "Bijoy", "sewing", "worker")
    add_employee("E016", "Nadia", "sewing", "worker")
    add_employee("E017", "Nadir", "sewing", "worker")
    add_employee("E018", "Kashem", "sewing", "worker")
    add_employee("E019", "Habiba", "sewing", "worker")
    add_employee("E020", "Fariha", "sewing", "worker")
    add_employee("E021", "Pasha", "sewing", "worker")
    add_employee("E022", "aria", "sewing", "worker")
    add_employee("E023", "sheuli", "sewing", "worker")
    add_employee("E024", "rina", "sewing", "worker")


